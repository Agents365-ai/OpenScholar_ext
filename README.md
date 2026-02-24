# OpenScholar Extension

This repository is forked from [AkariAsai/OpenScholar](https://github.com/AkariAsai/OpenScholar) with macOS local run support.

**English** | [中文](README_cn.md)

## macOS Local Run Guide

The original OpenScholar `run.py` requires vllm (Linux + CUDA only). This extension provides a macOS-compatible solution using **LM Studio** as the inference backend.

### Quick Start

1. **Install LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai/)
2. **Load Model**: Load `Llama-3.1-OpenScholar-8B` in LM Studio and start the server
3. **Set API Key** (optional):
   ```bash
   export S2_API_KEY=YOUR_SEMANTIC_SCHOLAR_API_KEY
   ```

### Pipeline Modes Comparison

| Mode | Retrieval | Reranking | Generation | Self-Feedback | Use Case |
|------|-----------|-----------|------------|---------------|----------|
| **Single Query** | S2 API realtime | ✗ | ✓ | ✗ | Quick single question test |
| **Standard RAG** | From input ctxs | ✗ | ✓ | ✗ | Pre-retrieved contexts |
| **Retriever + Reranker** | From input ctxs | ✓ FlagReranker | ✓ | ✗ | Higher relevance needed |
| **Self-Reflective** | From input ctxs + S2 | ✓ FlagReranker | ✓ | ✓ | Highest quality output |

**Quality vs Speed:**
```
Quality: Single Query < Standard RAG < Reranker < Self-Reflective
Speed:   Single Query > Standard RAG > Reranker > Self-Reflective
```

**Reranker Note:** Uses FlagReranker (priority) or CrossEncoder (fallback if FlagEmbedding not installed)

### OpenScholar Pipeline Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────┐    ┌──────────────────────────┐
│ ① Datastore │───▶│ ② Retriever │───▶│ ② Reranker  │───▶│  ③ LM   │───▶│ ④ Self-feedback Loop     │
│             │    │             │    │             │    │         │    │                          │
│ 45M papers  │    │ 240M embed  │    │ Top N docs  │    │ Generate│    │ Refine → Verify → y*     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────┘    └──────────────────────────┘
     │                   │                  │                 │                    │
     ▼                   ▼                  ▼                 ▼                    ▼
  Papers DB        Retrieve passages   Rerank by score   Initial answer    Iterative improvement
```

| Stage | Component | Description | run_openscholar.py |
|-------|-----------|-------------|-------------------|
| ① | **Datastore** | 45M papers, 240M embeddings | S2 API (`--ss_retriever`) or input `ctxs` |
| ② | **Retriever** | Retrieve initial passages | S2 API or pre-retrieved in input file |
| ② | **Reranker** | Rerank Top N by relevance score | `--ranking_ce` + `--reranker` |
| ③ | **LM** | Generate initial response y₀ | LM Studio API |
| ④ | **Self-feedback** | f₂: feedback → y*: refined answer | `--feedback` |

**Full Pipeline Flow:**
1. **Input**: Query x (e.g., "What are recent advancements in fluorescence biosensing?")
2. **Retrieve**: Get candidate passages from datastore
3. **Rerank**: Score passages (0.9, 0.5, 0.2...) → select Top N
4. **Generate**: LM produces initial response y₀ with citations c₀
5. **Feedback**: Generate feedback f₂ (e.g., "add more empirical findings...")
6. **Refine**: Produce improved response y* with verified citations c*

### Usage Examples

#### Single Query Mode (with S2 Retrieval)
- **Input**: Single question
- **Flow**: S2 retrieval → Generate answer
- **Best for**: Quick testing
```bash
python run_openscholar.py \
    -q "What is CRISPR gene editing?" \
    --ss_retriever \
    --top_n 10
```

#### Standard RAG Pipeline
- **Input**: JSON file with `ctxs`
- **Flow**: Use top_n ctxs directly → Generate answer
- **Best for**: When retrieval quality is already good

```bash
python run_openscholar.py \
    --input_file YOUR_INPUT_FILE \
    --use_contexts \
    --output_file output.json \
    --top_n 10 --llama3 --zero_shot
```

#### Retriever + Reranker Pipeline
- **Input**: JSON file with `ctxs`
- **Flow**: ctxs → Rerank by relevance → Generate answer
- **Best for**: Improving retrieval quality with reranking

```bash
python run_openscholar.py \
    --input_file YOUR_INPUT_FILE \
    --use_contexts --ranking_ce \
    --reranker BAAI/bge-reranker-base \
    --output_file output.json \
    --top_n 10 --llama3 --zero_shot
```

#### Self-Reflective Generation Pipeline
- **Input**: JSON file with `ctxs`
- **Flow**: ctxs → Rerank → Generate → Self-feedback → Improved answer
- **Best for**: Highest quality (2x LLM calls)

```bash
python run_openscholar.py \
    --input_file YOUR_INPUT_FILE \
    --use_contexts --ranking_ce \
    --reranker BAAI/bge-reranker-base \
    --feedback --use_abstract --norm_cite \
    --max_per_paper 3 \
    --output_file output.json \
    --top_n 10 --llama3 --zero_shot
```

### Advanced Configuration Options

| Option | Parameter | Description |
|--------|-----------|-------------|
| **Task-specific** | `--task_name scifact/pubmedqa/qasa` | Use task-specific prompt templates |
| **Citation filter** | `--min_citation 10` | Only keep papers with ≥10 citations |
| **Citation normalization** | `--norm_cite` | Normalize citation counts for ranking |
| **Per-paper limit** | `--max_per_paper 3` | Avoid single paper dominating results |
| **Post-hoc attribution** | `--posthoc_at` | Add citation attribution after generation |
| **Debug mode** | `--skip_generation` | Only retrieve/rerank, skip LLM generation |
| **Data sampling** | `--sample_k 100` | Randomly sample 100 items from dataset |
| **Resume from index** | `--start_index 50` | Start processing from item 50 |

### More Usage Examples

#### Task-Specific: SciFact (Claim Verification)
**Use case**: Verify if a scientific claim is true or false based on literature evidence.
- Input: A claim like "Vitamin C prevents cancer"
- Output: SUPPORTS / REFUTES / NOT ENOUGH INFO with citations
- Best for: Fact-checking scientific statements, paper review, news verification

```bash
python run_openscholar.py \
    --input_file scifact_data.json \
    --use_contexts --ranking_ce \
    --task_name scifact \
    --output_file scifact_output.json \
    --top_n 10 --llama3 --zero_shot
```

#### Task-Specific: PubMedQA (Yes/No/Maybe)
**Use case**: Answer biomedical yes/no questions based on PubMed abstracts.
- Input: Questions like "Is metformin effective for diabetes?"
- Output: Yes / No / Maybe with explanation
- Best for: Clinical decision support, medical literature QA

```bash
python run_openscholar.py \
    --input_file pubmedqa_data.json \
    --use_contexts --ranking_ce \
    --task_name pubmedqa \
    --output_file pubmedqa_output.json \
    --top_n 10 --llama3 --zero_shot
```

#### Task-Specific: QASA (Detailed QA)
**Use case**: Provide detailed answers requiring synthesis of multiple papers.
- Input: Open-ended questions like "What are the mechanisms of CRISPR?"
- Output: Comprehensive answer with multiple citations
- Best for: Literature review, research synthesis, educational content

```bash
python run_openscholar.py \
    --input_file qasa_data.json \
    --use_contexts --ranking_ce \
    --task_name qasa \
    --output_file qasa_output.json \
    --top_n 10 --llama3 --zero_shot
```

#### High-Quality Papers Only (Citation Filter)
**Use case**: Only use highly-cited papers as sources for more authoritative answers.
- `--min_citation 50`: Filters out papers with < 50 citations
- Best for: When you need well-established, peer-validated information
- Trade-off: May miss recent breakthrough papers (low citations due to recency)

```bash
python run_openscholar.py \
    -q "What are the latest treatments for Alzheimer's?" \
    --ss_retriever \
    --min_citation 50 \
    --top_n 10
```

#### Full Pipeline with All Optimizations
**Use case**: Maximum quality output for important queries.
- `--ranking_ce`: Rerank for relevance
- `--feedback`: Self-reflective improvement
- `--posthoc_at`: Add citation attribution
- `--use_abstract`: Use abstracts in reranking
- `--norm_cite`: Normalize citations for fair comparison
- `--min_citation 10`: Filter low-quality papers
- `--max_per_paper 3`: Diverse sources
- Trade-off: Slower (2x LLM calls), higher API cost

```bash
python run_openscholar.py \
    --input_file data.json \
    --use_contexts --ranking_ce \
    --reranker BAAI/bge-reranker-base \
    --feedback --posthoc_at \
    --use_abstract --norm_cite \
    --min_citation 10 \
    --max_per_paper 3 \
    --output_file best_quality.json \
    --top_n 10 --llama3 --zero_shot
```

#### Debug: Check Retrieval Quality (Skip Generation)
**Use case**: Evaluate retrieval and reranking without LLM generation.
- `--skip_generation`: Only outputs reranked ctxs, no LLM answer
- Best for: Debugging retrieval pipeline, evaluating reranker quality
- Output: JSON with reranked contexts, no "output" field

```bash
python run_openscholar.py \
    --input_file data.json \
    --use_contexts --ranking_ce \
    --skip_generation \
    --output_file retrieval_only.json \
    --top_n 10
```

#### Batch Processing with Sampling
**Use case**: Test pipeline on a random subset before full run.
- `--dataset`: Load directly from HuggingFace datasets
- `--sample_k 100`: Randomly sample 100 items
- Best for: Pipeline testing, hyperparameter tuning, cost estimation

```bash
python run_openscholar.py \
    --dataset OpenScholar/ScholarQABench \
    --sample_k 100 \
    --use_contexts --ranking_ce \
    --output_file sampled_output.json \
    --top_n 10 --llama3 --zero_shot
```

#### Resume Interrupted Processing
**Use case**: Continue from where you left off after interruption.
- `--start_index 500`: Skip first 500 items
- Auto-resume: If output file exists, automatically continues from last result
- Best for: Large batch processing, recovering from crashes

```bash
python run_openscholar.py \
    --input_file large_data.json \
    --start_index 500 \
    --use_contexts \
    --output_file output.json \
    --top_n 10 --llama3 --zero_shot
```

#### Reverse Order Processing
**Use case**: Process data from end to beginning.
- `--reverse`: Useful when you want to prioritize recent/later items
- Can combine with `--start_index` for flexible batch control

```bash
python run_openscholar.py \
    --input_file data.json \
    --reverse \
    --use_contexts \
    --output_file output.json \
    --top_n 10 --llama3 --zero_shot
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--ss_retriever` | Use Semantic Scholar API for paper retrieval |
| `--ranking_ce` | Enable cross-encoder reranking |
| `--feedback` | Enable self-reflective feedback loop |
| `--task_name` | Task type: `default`, `scifact`, `pubmedqa`, `qasa` |
| `--min_citation` | Minimum citation count filter |
| `--use_abstract` | Use abstract for reranking |
| `--max_per_paper` | Max passages per paper |

### Visualize Results

```bash
python visualize_results.py output.json --html --md --pdf
```

### Comparison: run.py vs run_openscholar.py

| Feature | run.py (Original) | run_openscholar.py (macOS) |
|---------|-------------------|---------------------------|
| **Platform** | Linux + CUDA | macOS / Linux / Windows |
| **Inference Backend** | vllm | LM Studio (OpenAI API) |
| **GPU Required** | Yes (NVIDIA) | No (Apple Silicon / CPU) |
| **Reranker** | FlagReranker | FlagReranker / CrossEncoder |
| **API Support** | OpenAI, Together, Anyscale | LM Studio local API |
| **S2 Retrieval** | Yes | Yes |
| **Self-reflective** | Yes | Yes |
| **All CLI params** | Full | Full compatible |

**When to use which:**
- `run.py`: Production deployment on Linux servers with NVIDIA GPUs
- `run_openscholar.py`: Local development on macOS, testing, or non-CUDA environments

### Parameter Comparison

| Parameter | run.py | run_openscholar.py | Description |
|-----------|--------|-------------------|-------------|
| `--input_file` | ✓ | ✓ | Input JSON/JSONL file |
| `--output_file` | ✓ | ✓ | Output file path |
| `--query` | ✗ | ✓ | Single query mode |
| `--task_name` | ✓ | ✓ | Task: default/scifact/pubmedqa/qasa |
| `--dataset` | ✓ | ✓ | HuggingFace dataset path |
| `--model_name` | ✓ | ✓ | Model name (run.py: actual, run_openscholar: display only) |
| `--use_contexts` | ✓ | ✓ | Use contexts from input |
| `--llama3` | ✓ | ✓ | Llama3 chat template |
| `--zero_shot` | ✓ | ✓ | Zero-shot inference |
| `--top_n` | ✓ | ✓ | Number of passages (default: 10) |
| `--max_tokens` | ✓ | ✓ | Max generation tokens |
| `--feedback` | ✓ | ✓ | Self-reflective feedback |
| `--posthoc_at` | ✓ | ✓ | Post-hoc attribution |
| `--ranking_ce` | ✓ | ✓ | Cross-encoder reranking |
| `--reranker` | ✓ | ✓ | Reranker model path |
| `--min_citation` | ✓ | ✓ | Min citation filter |
| `--norm_cite` | ✓ | ✓ | Normalize citations |
| `--ss_retriever` | ✓ | ✓ | Semantic Scholar retrieval |
| `--use_abstract` | ✓ | ✓ | Use abstract for reranking |
| `--max_per_paper` | ✓ | ✓ | Max passages per paper |
| `--skip_generation` | ✓ | ✓ | Skip LLM generation |
| `--sample_k` | ✓ | ✓ | Sample K items |
| `--reverse` | ✓ | ✓ | Reverse data order |
| `--start_index` | ✓ | ✓ | Start from index |
| `--api` | ✓ | ✓* | API provider (ignored, uses LM Studio) |
| `--api_key_fp` | ✓ | ✓* | API key file path (ignored, uses LM Studio) |
| `--download_dir` | ✓ | ✓* | Model download directory (ignored, uses LM Studio) |
| `--use_slow_tokenizer` | ✓ | ✓* | Slow tokenizer (ignored, uses LM Studio) |

*These parameters are accepted for CLI compatibility but ignored (LM Studio handles model loading)

---

## OpenScholar (Original)

[**Blog**](https://allenai.org/blog/openscholar) | [**Demo**](https://open-scholar.allen.ai/) | [**Paper**](https://arxiv.org/abs/2411.14199) | [**Model checkpoints and data**](https://huggingface.co/collections/OpenScholar/openscholar-v1-67376a89f6a80f448da411a6) | [**ScholarQABench**](https://github.com/AkariAsai/ScholarQABench/) | [**Expert Evaluation**](https://github.com/AkariAsai/OpenScholar_ExpertEval)

### Overview

Scientific progress hinges on our ability to find, synthesize, and build on relevant knowledge from the scientific literature. **OpenScholar** is a retrieval-augmented language model (LM) designed to answer user queries by first searching for relevant papers in the literature and then generating responses grounded in those sources.

![Overview of OpenScholar](imgs/open_scholar.png)

### Repository Structure

- [`src/`](src): Main source codes for OpenScholar
- [`training/`](training): Training code for Llama 3.1 8B
- [`retriever/`](retriever): Retrieval offline & online server code
- [`run_openscholar.py`](run_openscholar.py): macOS-compatible inference script (uses LM Studio)

### Installation

```bash
conda create -n os_env python=3.10.0
conda activate os_env
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Set API keys:
```bash
export S2_API_KEY=YOUR_S2_API_KEY
```

### Configuration Details

- `top_n`: Number of passages fed into the LM (default: 10)
- `feedback`: Enable self-feedback loop during generation
- `posthoc_at`: Enable posthoc citation attribution
- `zero_shot`: Zero-shot inference mode
- `ranking_ce`: Use reranking model for top_n passages
- `reranker`: Path to reranker model (local or HF hub)
- `min_citation`: Minimum citation count filter
- `ss_retriever`: Use Semantic Scholar API in feedback loop
- `use_abstract`: Use abstract for reranking
- `max_per_paper`: Max passages per paper
- `task_name`: Task name (SciFact: `claim_full`, PubmedQA: `boolean_question_full`, QASA: `single_qa`)

### Citation

```
@article{openscholar,
  title={{OpenScholar}: Synthesizing Scientific Literature with Retrieval-Augmented Language Models},
  author={Asai, Akari and He*, Jacqueline and Shao*, Rulin and Shi, Weijia and Singh, Amanpreet and Chang, Joseph Chee and Lo, Kyle and Soldaini, Luca and others},
  journal={Arxiv},
  year={2024},
}
```
