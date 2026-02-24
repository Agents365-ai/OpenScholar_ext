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

### Usage Examples

#### Single Query Mode (with S2 Retrieval)
```bash
python run_openscholar.py \
    -q "What is CRISPR gene editing?" \
    --ss_retriever \
    --top_n 10
```

#### Standard RAG Pipeline
```bash
python run_openscholar.py \
    --input_file YOUR_INPUT_FILE \
    --use_contexts \
    --output_file output.json \
    --top_n 10 --llama3 --zero_shot
```

#### Retriever + Reranker Pipeline
```bash
python run_openscholar.py \
    --input_file YOUR_INPUT_FILE \
    --use_contexts --ranking_ce \
    --reranker BAAI/bge-reranker-base \
    --output_file output.json \
    --top_n 10 --llama3 --zero_shot
```

#### Self-Reflective Generation Pipeline
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
python test_on_macos/visualize_results.py \
    output.json \
    --html --md --pdf
```

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
