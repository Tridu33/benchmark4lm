# RAG Application Performance & Quality Monitor

**LangFuse + Arize Phoenix** integration for RAG (Retrieval-Augmented Generation) data drift and quality monitoring.

## Supported Frameworks

| Framework | Description |
|-----------|-------------|
| **LangChain** | LangChain RAG pipelines with FAISS/Chroma/Pinecone |
| **LlamaIndex** | LlamaIndex data frameworks |

## Observability Stack

```
┌─────────────────────────────────────────────────────┐
│                    RAG Pipeline                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Embedding│→ │ Retrieval│→ │ LLM Generation    │  │
│  │ (Vector) │  │ (Top-K)  │  │ (Answer Synthesis) │  │
│  └──────────┘  └──────────┘  └───────────────────┘  │
└──────┬──────────────┬───────────────┬────────────────┘
       │              │               │
  ┌────▼────┐   ┌─────▼────┐   ┌─────▼─────┐
  │LangFuse │   │ Phoenix  │   │  RAGAS    │
  │Tracing  │   │  Drift   │   │ Eval      │
  │+ Cost   │   │Detection │   │ Metrics   │
  └─────────┘   └──────────┘   └───────────┘
```

## Monitored Metrics

### Retrieval
- **Retrieval latency**: Embedding + vector search time
- **Top-K relevance**: Retrieved document count and quality
- **Context precision**: How well retrieved docs match the question

### Generation
- **Generation latency**: LLM response time
- **Answer faithfulness**: Does the answer match the retrieved context?
- **Answer relevancy**: Does the answer address the question?
- **Context recall**: Does retrieved context contain the ground truth?

### Data Drift (Phoenix)
- **Embedding distribution shifts**: PCA/UMAP visualization
- **Query drift**: Distribution of incoming queries vs training data
- **Document drift**: Knowledge base changes over time

### Cost (LangFuse)
- **Token usage**: Prompt + completion tokens per query
- **Cost tracking**: Per-query and aggregate cost
- **Trace analysis**: Full request/response traces

## Quick Start

```bash
# LangChain RAG evaluation
python profile_rag.py \
    --framework langchain \
    --embedding-model text-embedding-3-small \
    --llm-model gpt-4o \
    --eval-dataset eval_questions.jsonl \
    --out-dir ./out \
    --langfuse \
    --phoenix \
    --ragas

# LlamaIndex RAG evaluation
python profile_rag.py \
    --framework llamaindex \
    --out-dir ./out

# Via task script
bash run_rag.sh langchain gpt-4o text-embedding-3-small eval_questions.jsonl 50
```

## Eval Dataset Format (JSONL)

```jsonl
{"question": "What is RAG?", "ground_truth": "RAG combines retrieval with generation."}
{"question": "How does FAISS work?", "ground_truth": "FAISS uses IVF/HNSW for ANN search."}
```

## RAG Quality Patterns

RAG applications face unique challenges:
- **Retrieval bottleneck**: Poor retrieval quality limits answer quality
- **Context window limits**: Too many retrieved docs exceed LLM context
- **Data freshness**: Outdated documents produce outdated answers
- **Hallucination**: LLM may ignore retrieved context
- **Embedding drift**: Model updates change embedding distributions
- **Cost scaling**: Embedding + LLM calls multiply per query
