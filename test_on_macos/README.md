# OpenScholar on macOS

This directory contains scripts for running OpenScholar pipelines on macOS without CUDA/vllm dependencies.

## Prerequisites

1. **LM Studio** - Download from [lmstudio.ai](https://lmstudio.ai)
2. Load the model: `OpenScholar/Llama-3.1_OpenScholar-8B`
3. Start the local server (default: `http://localhost:1234`)

## Scripts

### run_lmstudio.py
Basic RAG pipeline using LM Studio for generation.

```bash
python run_lmstudio.py \
    --input_file test_input.json \
    --output_file output.json \
    --top_n 10
```

### run_reranker.py
Full Retriever + Reranker pipeline with CrossEncoder reranking.

```bash
python run_reranker.py \
    --input_file test_input.json \
    --output_file output.json \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --top_n 10
```

### visualize_results.py
Render results to HTML/PDF format.

```bash
python visualize_results.py \
    --input_file test_output.json \
    --output results
```

## Input Format

```json
{
  "data": [
    {
      "input": "Your question here",
      "ctxs": [
        {"title": "Paper Title", "text": "Retrieved passage..."}
      ]
    }
  ]
}
```

## Dependencies

```bash
pip install requests tqdm sentence-transformers
```

## Comparison with Original run.py

| Feature | run.py (Linux) | run_reranker.py (macOS) |
|---------|----------------|-------------------------|
| Inference | vllm (CUDA) | LM Studio API |
| Reranker | FlagReranker | CrossEncoder |
| Hardware | GPU required | CPU/Apple Silicon |
