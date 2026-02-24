# OpenScholar Extension

本仓库 fork 自 [AkariAsai/OpenScholar](https://github.com/AkariAsai/OpenScholar)，添加了 macOS 本地运行支持。

[**English**](README.md) | **中文**

## macOS 本地运行指南

原版 OpenScholar 的 `run.py` 依赖 vllm，仅支持 Linux + CUDA 环境。本扩展提供了 macOS 兼容方案，使用 **LM Studio** 作为推理后端。

### 快速开始

1. **安装 LM Studio**：从 [lmstudio.ai](https://lmstudio.ai/) 下载并安装
2. **加载模型**：在 LM Studio 中加载 `Llama-3.1-OpenScholar-8B` 模型并启动服务器
3. **设置 API Key**（可选）：
   ```bash
   export S2_API_KEY=你的Semantic_Scholar_API_KEY
   ```

### 使用示例

#### 单问题模式（使用 S2 检索）
```bash
python run_openscholar.py \
    -q "What is CRISPR gene editing?" \
    --ss_retriever \
    --top_n 10
```

#### 标准 RAG Pipeline
```bash
python run_openscholar.py \
    --input_file YOUR_INPUT_FILE \
    --use_contexts \
    --output_file output.json \
    --top_n 10 --llama3 --zero_shot
```

#### 检索 + 重排序 Pipeline
```bash
python run_openscholar.py \
    --input_file YOUR_INPUT_FILE \
    --use_contexts --ranking_ce \
    --reranker BAAI/bge-reranker-base \
    --output_file output.json \
    --top_n 10 --llama3 --zero_shot
```

#### 自反馈生成 Pipeline
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

### 参数说明

| 参数 | 说明 |
|------|------|
| `--ss_retriever` | 使用 Semantic Scholar API 进行论文检索 |
| `--ranking_ce` | 启用交叉编码器重排序 |
| `--feedback` | 启用自反馈生成循环 |
| `--task_name` | 任务类型：`default`, `scifact`, `pubmedqa`, `qasa` |
| `--min_citation` | 最小引用数过滤 |
| `--use_abstract` | 使用摘要进行重排序 |
| `--max_per_paper` | 每篇论文最大段落数 |

### 结果可视化

```bash
python test_on_macos/visualize_results.py \
    output.json \
    --html --md --pdf
```

---

## OpenScholar 概述

科学进步依赖于我们从科学文献中查找、综合和利用相关知识的能力。然而，文献的指数级增长——每年发表数百万篇论文——使科学家越来越难以找到所需信息，甚至难以跟上单个子领域的最新发现。

为帮助科学家有效地浏览和综合科学文献，我们推出了 **OpenScholar**，这是一个检索增强语言模型（LM），旨在通过首先在文献中搜索相关论文，然后生成基于这些来源的回答来回答用户查询。

![OpenScholar 概述](imgs/open_scholar.png)

## 仓库结构

- [`src/`](src): OpenScholar 主要源代码
- [`training/`](training): 使用处理后数据训练 Llama 3.1 8B 的训练代码
- [`retriever/`](retriever): 离线检索和在线检索服务器代码
- [`run_openscholar.py`](run_openscholar.py): macOS 兼容的推理脚本（使用 LM Studio）

## 安装

```bash
conda create -n os_env python=3.10.0
conda activate os_env
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

设置 API Key：
```bash
export S2_API_KEY=YOUR_S2_API_KEY
```

## 配置详解

- `top_n`: 输入到 LM 的段落数量，默认为 10
- `feedback`: 启用生成过程中的自反馈循环
- `posthoc_at`: 启用后置引用归因
- `zero_shot`: 零样本推理模式
- `ranking_ce`: 使用重排序模型对 top_n 段落重新排序
- `reranker`: 重排序模型路径（本地或 HF hub）
- `min_citation`: 最小引用数过滤
- `ss_retriever`: 在反馈生成循环中使用 Semantic Scholar API
- `use_abstract`: 使用摘要增强重排序效果
- `max_per_paper`: 每篇论文的最大段落数
- `task_name`: 任务名称（SciFact: `claim_full`, PubmedQA: `boolean_question_full`, QASA: `single_qa`）

## 引用

```
@article{openscholar,
  title={{OpenScholar}: Synthesizing Scientific Literature with Retrieval-Augmented Language Models},
  author={Asai, Akari and He*, Jacqueline and Shao*, Rulin and Shi, Weijia and Singh, Amanpreet and Chang, Joseph Chee and Lo, Kyle and Soldaini, Luca and others},
  journal={Arxiv},
  year={2024},
}
```
