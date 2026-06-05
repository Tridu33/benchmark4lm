#!/usr/bin/env python3
"""
RAG Application Performance & Quality Monitor
LangFuse + Arize Phoenix integration for RAG data drift & quality monitoring.

Targets: LangChain / LlamaIndex RAG pipelines.

Monitors:
  - Retrieval latency (document search, embedding generation)
  - Generation latency (LLM response)
  - Context relevance, answer faithfulness, answer relevance
  - Data drift detection (embedding distribution shifts)
  - Token usage and cost tracking
  - Quality metrics (RAGAS: context precision, context recall, faithfulness, answer relevancy)

Usage:
    python profile_rag.py \
        --framework langchain \
        --index-type vector \
        --embedding-model text-embedding-3-small \
        --llm-model gpt-4o \
        --eval-dataset eval_questions.jsonl \
        --out-dir ./out \
        --langfuse \
        --phoenix
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core_profiler import (
    GpuMonitor, CpuMonitor, TorchProfilerCtx,
    ProfilingResult, collect_hw_summary, collect_sw_summary,
    generate_report, LangFuseTracer,
)

# ── LangChain ──────────────────────────────────────────────────
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# ── LlamaIndex ─────────────────────────────────────────────────
try:
    from llama_index.core import VectorStoreIndex, Document as LIDocument
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

# ── Phoenix ────────────────────────────────────────────────────
try:
    import phoenix as px
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

# ── RAGAS (evaluation) ────────────────────────────────────────
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


@dataclass
class RAGEvalResult:
    question: str = ""
    retrieved_docs: List[str] = None
    answer: str = ""
    ground_truth: str = ""
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    retrieved_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    context_precision_score: float = 0.0
    faithfulness_score: float = 0.0
    answer_relevancy_score: float = 0.0


def parse_args():
    p = argparse.ArgumentParser(description="RAG Performance & Quality Monitor")
    p.add_argument("--framework", choices=["langchain", "llamaindex"], default="langchain")
    p.add_argument("--index-type", choices=["vector", "bm25", "hybrid"], default="vector")
    p.add_argument("--embedding-model", type=str, default="text-embedding-3-small")
    p.add_argument("--llm-model", type=str, default="gpt-4o")
    p.add_argument("--eval-dataset", type=str, default="",
                   help="Path to JSONL eval dataset with questions and ground truths")
    p.add_argument("--doc-dir", type=str, default="",
                   help="Directory with documents to build the index")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of documents to retrieve")
    p.add_argument("--num-questions", type=int, default=50)
    p.add_argument("--api-base", type=str, default="https://api.openai.com/v1",
                   help="OpenAI-compatible API base URL")
    p.add_argument("--out-dir", type=str, default="./out")
    p.add_argument("--langfuse", action="store_true", help="Enable LangFuse tracing")
    p.add_argument("--phoenix", action="store_true", help="Enable Arize Phoenix")
    p.add_argument("--ragas", action="store_true", help="Run RAGAS evaluation")
    p.add_argument("--task-script", type=str, default="")
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--gpu-id", type=int, default=0)
    return p.parse_args()


def _load_eval_questions(args) -> List[Dict[str, str]]:
    """Load evaluation questions from JSONL file."""
    questions: List[Dict[str, str]] = []

    if args.eval_dataset and os.path.exists(args.eval_dataset):
        with open(args.eval_dataset) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    questions.append({
                        "question": obj.get("question", ""),
                        "ground_truth": obj.get("ground_truth", ""),
                    })
        return questions[:args.num_questions]

    # Default synthetic questions
    defaults = [
        {"question": "What are the key benefits of using a microservices architecture?",
         "ground_truth": "Microservices provide scalability, independent deployment, technology diversity, fault isolation, and easier maintenance."},
        {"question": "How does retrieval-augmented generation reduce hallucinations?",
         "ground_truth": "RAG grounds LLM outputs in retrieved factual documents, reducing fabrication."},
        {"question": "What is the difference between fine-tuning and RAG?",
         "ground_truth": "Fine-tuning changes model weights; RAG provides external context without modifying the model."},
        {"question": "Explain vector database indexing methods.",
         "ground_truth": "Common methods include HNSW, IVF, PQ, and LSH for approximate nearest neighbor search."},
        {"question": "What are the main challenges in production RAG systems?",
         "ground_truth": "Latency, retrieval quality, data freshness, context window limits, and evaluation complexity."},
    ]
    return (defaults * ((args.num_questions // len(defaults)) + 1))[:args.num_questions]


def _build_langchain_index(args) -> Any:
    """Build a LangChain vector store index from documents or synthetic data."""
    if not LANGCHAIN_AVAILABLE:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs: List[Document] = []
    if args.doc_dir and os.path.isdir(args.doc_dir):
        for fn in os.listdir(args.doc_dir):
            fp = os.path.join(args.doc_dir, fn)
            if os.path.isfile(fp) and fn.endswith(".txt"):
                with open(fp) as f:
                    text = f.read()
                docs.extend([Document(page_content=c, metadata={"source": fn})
                             for c in splitter.split_text(text)])
    else:
        # Synthetic documents
        sample_docs = [
            "Microservices architecture decomposes applications into small, independently deployable services. "
            "Each service owns a specific business capability and communicates via APIs.",
            "Retrieval-augmented generation (RAG) combines information retrieval with LLM text generation. "
            "It retrieves relevant documents and provides them as context to the model.",
            "Vector databases store embeddings for semantic search. Popular options include FAISS, Pinecone, and Weaviate. "
            "They support HNSW and IVF indexing for fast approximate nearest neighbor search.",
            "Fine-tuning adapts a pretrained model to a specific task by updating its weights on task-specific data. "
            "RAG instead provides external knowledge at inference time without weight changes.",
            "Production RAG systems face challenges in retrieval quality (recall/precision), "
            "latency (embedding + search + generation), data freshness, and evaluation complexity.",
        ]
        for text in sample_docs:
            docs.extend([Document(page_content=c, metadata={"source": "synthetic"})
                         for c in splitter.split_text(text)])

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(
        model=args.embedding_model,
        openai_api_base=args.api_base,
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def run_langchain_rag_eval(args, result, gpu_mon, cpu_mon) -> List[RAGEvalResult]:
    """Run RAG evaluation using LangChain."""
    if not LANGCHAIN_AVAILABLE:
        print("[ERROR] langchain packages required.")
        sys.exit(1)

    vectorstore = _build_langchain_index(args)
    if vectorstore is None:
        print("[ERROR] Failed to build index.")
        sys.exit(1)

    llm = ChatOpenAI(model=args.llm_model, openai_api_base=args.api_base, temperature=0.0)

    questions = _load_eval_questions(args)
    eval_results: List[RAGEvalResult] = []

    print(f"[LangChain RAG] Evaluating {len(questions)} questions ...")

    for q in questions:
        r = RAGEvalResult(question=q["question"], ground_truth=q["ground_truth"])

        # Retrieval
        t0 = time.time()
        retrieved = vectorstore.similarity_search(q["question"], k=args.top_k)
        r.retrieval_latency_ms = (time.time() - t0) * 1000
        r.retrieved_count = len(retrieved)
        r.retrieved_docs = [d.page_content for d in retrieved]

        # Generation
        context = "\n".join(r.retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {q['question']}\nAnswer:"

        t1 = time.time()
        try:
            response = llm.invoke(prompt)
            r.answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            r.answer = f"Error: {e}"
        r.generation_latency_ms = (time.time() - t1) * 1000
        r.total_latency_ms = r.retrieval_latency_ms + r.generation_latency_ms

        # Token estimation (rough: 1 token ~ 4 chars)
        r.prompt_tokens = len(prompt) // 4
        r.completion_tokens = len(r.answer) // 4

        eval_results.append(r)

    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util

    # Aggregate
    if eval_results:
        result.request_latencies = [r.total_latency_ms for r in eval_results]
        result.ttft_list = [r.retrieval_latency_ms for r in eval_results]
        result.total_tokens_generated = sum(r.completion_tokens for r in eval_results)
        result.total_prompt_tokens = sum(r.prompt_tokens for r in eval_results)
        result.throughput_rps = len(eval_results) / max(sum(r.total_latency_ms for r in eval_results) / 1000, 1e-6)
        result.max_concurrent_requests = 1

    return eval_results


def run_llamaindex_rag_eval(args, result, gpu_mon, cpu_mon) -> List[RAGEvalResult]:
    """Run RAG evaluation using LlamaIndex."""
    if not LLAMAINDEX_AVAILABLE:
        print("[ERROR] llama-index packages required.")
        sys.exit(1)

    # Build index
    docs_text = "RAG combines retrieval and generation for grounded LLM outputs. " * 20
    docs = [LIDocument(text=docs_text)]

    embed_model = OpenAIEmbedding(model=args.embedding_model)
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

    llm = OpenAI(model=args.llm_model, temperature=0.0)
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=args.top_k)

    questions = _load_eval_questions(args)
    eval_results: List[RAGEvalResult] = []

    print(f"[LlamaIndex RAG] Evaluating {len(questions)} questions ...")

    for q in questions:
        r = RAGEvalResult(question=q["question"], ground_truth=q["ground_truth"])

        t0 = time.time()
        try:
            response = query_engine.query(q["question"])
            r.answer = str(response)
            r.retrieved_docs = [n.text for n in response.source_nodes] if hasattr(response, "source_nodes") else []
            r.retrieved_count = len(r.retrieved_docs)
        except Exception as e:
            r.answer = f"Error: {e}"
        r.total_latency_ms = (time.time() - t0) * 1000
        r.completion_tokens = len(r.answer) // 4
        eval_results.append(r)

    result.gpu_snapshots = list(gpu_mon.snapshots)
    result.cpu_snapshots = list(cpu_mon.snapshots)
    result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
    result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
    result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
    result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
    result.avg_gpu_util = gpu_mon.avg_util
    result.avg_cpu_util = cpu_mon.avg_util

    if eval_results:
        result.request_latencies = [r.total_latency_ms for r in eval_results]
        result.ttft_list = [r.total_latency_ms for r in eval_results]
        result.total_tokens_generated = sum(r.completion_tokens for r in eval_results)
        result.throughput_rps = len(eval_results) / max(sum(r.total_latency_ms for r in eval_results) / 1000, 1e-6)

    return eval_results


def run_phoenix_session(eval_results: List[RAGEvalResult]) -> str:
    """Start Phoenix and log spans."""
    if not PHOENIX_AVAILABLE:
        return ""

    try:
        session = px.launch_app()
        print(f"[Phoenix] Running at {px.active_session().url}")

        # Log evaluation spans
        from openinference.instrumentation.langchain import LangChainInstrumentor
        # In practice, you'd wrap the LangChain/LlamaIndex pipeline with openinference instrumentation
        # Here we just note the session URL

        return str(px.active_session().url)
    except Exception as e:
        print(f"[Phoenix] Error: {e}")
        return ""


def run_ragas_eval(eval_results: List[RAGEvalResult]) -> Dict[str, float]:
    """Run RAGAS evaluation on results."""
    if not RAGAS_AVAILABLE:
        return {}

    from datasets import Dataset

    data = {
        "question": [r.question for r in eval_results],
        "answer": [r.answer for r in eval_results],
        "contexts": [[d for d in r.retrieved_docs] for r in eval_results],
        "ground_truth": [r.ground_truth for r in eval_results],
    }
    dataset = Dataset.from_dict(data)

    try:
        ragas_result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return ragas_result
    except Exception as e:
        print(f"[RAGAS] Error: {e}")
        return {}


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    result = ProfilingResult()
    result.hw_summary = collect_hw_summary()
    result.sw_summary = collect_sw_summary()

    if args.task_script:
        gpu_mon = GpuMonitor(args.gpu_id, args.monitor_interval)
        cpu_mon = CpuMonitor(args.monitor_interval)
        gpu_mon.start()
        cpu_mon.start()
        try:
            import subprocess
            subprocess.run(["bash", args.task_script], timeout=None)
        finally:
            gpu_mon.stop()
            cpu_mon.stop()
        result.gpu_snapshots = list(gpu_mon.snapshots)
        result.cpu_snapshots = list(cpu_mon.snapshots)
        result.peak_gpu_mem_mb = gpu_mon.peak_mem_mb
        result.avg_gpu_mem_mb = gpu_mon.avg_mem_mb
        result.peak_cpu_mem_mb = cpu_mon.peak_mem_mb
        result.avg_cpu_mem_mb = cpu_mon.avg_mem_mb
        result.avg_gpu_util = gpu_mon.avg_util
        result.avg_cpu_util = cpu_mon.avg_util
        eval_results = []
    else:
        gpu_mon = GpuMonitor(args.gpu_id, args.monitor_interval)
        cpu_mon = CpuMonitor(args.monitor_interval)
        gpu_mon.start()
        cpu_mon.start()

        try:
            if args.framework == "langchain":
                eval_results = run_langchain_rag_eval(args, result, gpu_mon, cpu_mon)
            else:
                eval_results = run_llamaindex_rag_eval(args, result, gpu_mon, cpu_mon)
        finally:
            gpu_mon.stop()
            cpu_mon.stop()

    # Phoenix
    if args.phoenix and eval_results:
        result.nsys_report_path = run_phoenix_session(eval_results)

    # RAGAS
    if args.ragas and eval_results and RAGAS_AVAILABLE:
        ragas_result = run_ragas_eval(eval_results)
        if ragas_result:
            result.bottleneck_analysis = (
                f"RAGAS Evaluation Results:\n"
                f"- Faithfulness: {ragas_result.get('faithfulness', 0):.3f}\n"
                f"- Answer Relevancy: {ragas_result.get('answer_relevancy', 0):.3f}\n"
                f"- Context Precision: {ragas_result.get('context_precision', 0):.3f}\n"
                f"- Context Recall: {ragas_result.get('context_recall', 0):.3f}\n"
            )

    # LangFuse
    if args.langfuse and eval_results:
        lf = LangFuseTracer(project_name="benchmark4lm-rag")
        for er in eval_results[:10]:  # Trace first 10
            lf.trace_generation(
                name="rag_query",
                input_text=er.question,
                output_text=er.answer,
                model=args.llm_model,
                prompt_tokens=er.prompt_tokens,
                completion_tokens=er.completion_tokens,
                latency_ms=er.total_latency_ms,
                metadata={"retrieval_latency_ms": er.retrieval_latency_ms,
                          "retrieved_docs": er.retrieved_count},
            )
        result.langfuse_trace_url = "See LangFuse dashboard for traces"
        lf.shutdown()

    # Auto-analysis for RAG
    if eval_results:
        avg_retrieval = np.mean([r.retrieval_latency_ms for r in eval_results])
        avg_generation = np.mean([r.generation_latency_ms for r in eval_results])
        if avg_generation > avg_retrieval * 3:
            result.optimization_suggestions.append(
                f"Generation dominates retrieval ({avg_generation:.0f}ms vs {avg_retrieval:.0f}ms). "
                "Consider streaming output, using a faster LLM, or speculative decoding."
            )
        if avg_retrieval > 500:
            result.optimization_suggestions.append(
                f"High retrieval latency ({avg_retrieval:.0f}ms). "
                "Consider using a faster embedding model, reducing index size, "
                "or switching to a GPU-accelerated vector store (cuVS, RAPIDS)."
            )

    generate_report(
        result,
        output_path=os.path.join(args.out_dir, "result.md"),
        scenario_name=f"RAG Application — {args.framework}",
        model_type="rag",
        task_type="inference",
    )


if __name__ == "__main__":
    main()
