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

### Pipeline 模式对比

| 模式 | 检索 | 重排序 | 生成 | 自反馈 | 适用场景 |
|------|------|--------|------|--------|----------|
| **Single Query** | S2 API 实时检索 | ✗ | ✓ | ✗ | 快速单问题测试 |
| **Standard RAG** | 使用输入文件 ctxs | ✗ | ✓ | ✗ | 离线检索已完成 |
| **Retriever + Reranker** | 使用输入文件 ctxs | ✓ FlagReranker | ✓ | ✗ | 需要更高相关性 |
| **Self-Reflective** | 使用输入文件 ctxs + S2 | ✓ FlagReranker | ✓ | ✓ | 最高质量输出 |

**质量 vs 速度：**
```
质量: Single Query < Standard RAG < Reranker < Self-Reflective
速度: Single Query > Standard RAG > Reranker > Self-Reflective
```

**重排序说明：** 优先使用 FlagReranker，如未安装 FlagEmbedding 则使用 CrossEncoder 备选

### OpenScholar Pipeline 架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────┐    ┌──────────────────────────┐
│ ① Datastore │───▶│ ② Retriever │───▶│ ② Reranker  │───▶│  ③ LM   │───▶│ ④ Self-feedback Loop     │
│   数据存储   │    │    检索器    │    │   重排序器   │    │ 语言模型 │    │      自反馈循环           │
│ 4500万论文  │    │ 2.4亿向量   │    │  Top N 文档  │    │  生成    │    │ 改进 → 验证 → y*         │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────┘    └──────────────────────────┘
     │                   │                  │                 │                    │
     ▼                   ▼                  ▼                 ▼                    ▼
   论文库            检索段落           按分数重排         初始答案            迭代改进
```

| 阶段 | 组件 | 说明 | run_openscholar.py 对应 |
|------|------|------|------------------------|
| ① | **Datastore** | 4500万论文，2.4亿向量 | S2 API (`--ss_retriever`) 或输入 `ctxs` |
| ② | **Retriever** | 检索初始段落 | S2 API 或输入文件中预检索 |
| ② | **Reranker** | 按相关性分数重排 Top N | `--ranking_ce` + `--reranker` |
| ③ | **LM** | 生成初始响应 y₀ | LM Studio API |
| ④ | **Self-feedback** | f₂: 反馈 → y*: 改进答案 | `--feedback` |

**完整 Pipeline 流程：**
1. **输入**：查询 x（如 "荧光生物传感的最新研究进展？"）
2. **检索**：从数据存储获取候选段落
3. **重排序**：对段落打分（0.9, 0.5, 0.2...）→ 选择 Top N
4. **生成**：LM 生成初始响应 y₀ 和引用 c₀
5. **反馈**：生成反馈 f₂（如 "添加更多实证研究..."）
6. **改进**：生成改进响应 y* 和验证后的引用 c*

### 使用示例

#### 单问题模式（使用 S2 检索）
- **输入**：单个问题
- **流程**：S2 检索 → 生成答案
- **适用**：快速测试
```bash
python run_openscholar.py \
    -q "What is CRISPR gene editing?" \
    --ss_retriever \
    --top_n 10
```

#### 标准 RAG Pipeline
- **输入**：包含 `ctxs` 的 JSON 文件
- **流程**：直接使用 top_n 个 ctxs → 生成答案
- **适用**：检索质量已足够好

```bash
python run_openscholar.py \
    --input_file YOUR_INPUT_FILE \
    --use_contexts \
    --output_file output.json \
    --top_n 10 --llama3 --zero_shot
```

#### 检索 + 重排序 Pipeline
- **输入**：包含 `ctxs` 的 JSON 文件
- **流程**：ctxs → 重排序选出最相关 → 生成答案
- **适用**：通过重排序提升检索质量

```bash
python run_openscholar.py \
    --input_file YOUR_INPUT_FILE \
    --use_contexts --ranking_ce \
    --reranker BAAI/bge-reranker-base \
    --output_file output.json \
    --top_n 10 --llama3 --zero_shot
```

#### 自反馈生成 Pipeline
- **输入**：包含 `ctxs` 的 JSON 文件
- **流程**：ctxs → 重排序 → 生成 → 自我反馈 → 改进生成
- **适用**：最高质量输出（2次 LLM 调用）

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

### 高级配置选项

| 选项 | 参数 | 说明 |
|------|------|------|
| **任务特定** | `--task_name scifact/pubmedqa/qasa` | 使用任务特定的 prompt 模板 |
| **引用数过滤** | `--min_citation 10` | 只保留引用数 ≥10 的论文 |
| **引用归一化** | `--norm_cite` | 归一化引用数用于排序 |
| **每篇论文限制** | `--max_per_paper 3` | 避免单篇论文主导结果 |
| **后置归因** | `--posthoc_at` | 生成后添加引用归因 |
| **调试模式** | `--skip_generation` | 只检索/重排序，跳过 LLM 生成 |
| **数据采样** | `--sample_k 100` | 从数据集随机采样 100 条 |
| **断点续传** | `--start_index 50` | 从第 50 条开始处理 |

### 更多使用示例

#### 任务特定：SciFact（声明验证）
**使用场景**：验证一个科学声明是否正确。
- 输入：声明如 "维生素C可以预防癌症"
- 输出：SUPPORTS（支持）/ REFUTES（反驳）/ NOT ENOUGH INFO（信息不足）+ 引用
- 适用：科学声明事实核查、论文审稿、新闻验证

```bash
python run_openscholar.py \
    --input_file scifact_data.json \
    --use_contexts --ranking_ce \
    --task_name scifact \
    --output_file scifact_output.json \
    --top_n 10 --llama3 --zero_shot
```

#### 任务特定：PubMedQA（是/否/可能）
**使用场景**：基于 PubMed 摘要回答生物医学是非问题。
- 输入：问题如 "二甲双胍对糖尿病有效吗？"
- 输出：Yes / No / Maybe + 解释
- 适用：临床决策支持、医学文献问答

```bash
python run_openscholar.py \
    --input_file pubmedqa_data.json \
    --use_contexts --ranking_ce \
    --task_name pubmedqa \
    --output_file pubmedqa_output.json \
    --top_n 10 --llama3 --zero_shot
```

#### 任务特定：QASA（详细问答）
**使用场景**：提供需要综合多篇论文的详细答案。
- 输入：开放式问题如 "CRISPR 的工作机制是什么？"
- 输出：包含多个引用的综合答案
- 适用：文献综述、研究综合、教育内容

```bash
python run_openscholar.py \
    --input_file qasa_data.json \
    --use_contexts --ranking_ce \
    --task_name qasa \
    --output_file qasa_output.json \
    --top_n 10 --llama3 --zero_shot
```

#### 仅高质量论文（引用数过滤）
**使用场景**：只使用高引用论文作为来源，获得更权威的答案。
- `--min_citation 50`：过滤引用数 < 50 的论文
- 适用：需要成熟、经过同行验证的信息
- 权衡：可能错过最新突破性论文（因新发表引用数低）

```bash
python run_openscholar.py \
    -q "What are the latest treatments for Alzheimer's?" \
    --ss_retriever \
    --min_citation 50 \
    --top_n 10
```

#### 完整 Pipeline（所有优化）
**使用场景**：重要查询的最高质量输出。
- `--ranking_ce`：重排序提升相关性
- `--feedback`：自反馈改进
- `--posthoc_at`：添加引用归因
- `--use_abstract`：使用摘要进行重排序
- `--norm_cite`：归一化引用数以公平比较
- `--min_citation 10`：过滤低质量论文
- `--max_per_paper 3`：来源多样化
- 权衡：速度较慢（2次 LLM 调用），API 成本更高

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

#### 调试：检查检索质量（跳过生成）
**使用场景**：评估检索和重排序效果，不进行 LLM 生成。
- `--skip_generation`：只输出重排序后的 ctxs，无 LLM 答案
- 适用：调试检索流程、评估重排序器质量
- 输出：JSON 包含重排序后的上下文，无 "output" 字段

```bash
python run_openscholar.py \
    --input_file data.json \
    --use_contexts --ranking_ce \
    --skip_generation \
    --output_file retrieval_only.json \
    --top_n 10
```

#### 批量处理（采样）
**使用场景**：在完整运行前先用随机子集测试流程。
- `--dataset`：直接从 HuggingFace datasets 加载
- `--sample_k 100`：随机采样 100 条
- 适用：流程测试、超参数调优、成本估算

```bash
python run_openscholar.py \
    --dataset OpenScholar/ScholarQABench \
    --sample_k 100 \
    --use_contexts --ranking_ce \
    --output_file sampled_output.json \
    --top_n 10 --llama3 --zero_shot
```

#### 断点续传
**使用场景**：中断后从上次位置继续处理。
- `--start_index 500`：跳过前 500 条
- 自动续传：如果输出文件存在，自动从最后结果继续
- 适用：大批量处理、崩溃恢复

```bash
python run_openscholar.py \
    --input_file large_data.json \
    --start_index 500 \
    --use_contexts \
    --output_file output.json \
    --top_n 10 --llama3 --zero_shot
```

#### 反向顺序处理
**使用场景**：从后往前处理数据。
- `--reverse`：适用于优先处理较新/靠后的数据
- 可与 `--start_index` 组合实现灵活的批次控制

```bash
python run_openscholar.py \
    --input_file data.json \
    --reverse \
    --use_contexts \
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
python visualize_results.py output.json --html --md --pdf
```

### 功能对比：run.py vs run_openscholar.py

| 功能 | run.py (原版) | run_openscholar.py (macOS) |
|------|--------------|---------------------------|
| **平台** | Linux + CUDA | macOS / Linux / Windows |
| **推理后端** | vllm | LM Studio (OpenAI API) |
| **需要GPU** | 是 (NVIDIA) | 否 (Apple Silicon / CPU) |
| **重排序器** | FlagReranker | FlagReranker / CrossEncoder |
| **API支持** | OpenAI, Together, Anyscale | LM Studio 本地 API |
| **S2检索** | 支持 | 支持 |
| **自反馈生成** | 支持 | 支持 |
| **CLI参数** | 完整 | 完全兼容 |

**使用场景：**
- `run.py`：在配备 NVIDIA GPU 的 Linux 服务器上进行生产部署
- `run_openscholar.py`：macOS 本地开发、测试或无 CUDA 环境

### 参数对比

| 参数 | run.py | run_openscholar.py | 说明 |
|------|--------|-------------------|------|
| `--input_file` | ✓ | ✓ | 输入 JSON/JSONL 文件 |
| `--output_file` | ✓ | ✓ | 输出文件路径 |
| `--query` | ✗ | ✓ | 单问题模式 |
| `--task_name` | ✓ | ✓ | 任务类型：default/scifact/pubmedqa/qasa |
| `--dataset` | ✓ | ✓ | HuggingFace 数据集路径 |
| `--model_name` | ✓ | ✓ | 模型名称（run.py：实际加载，run_openscholar：仅显示） |
| `--use_contexts` | ✓ | ✓ | 使用输入中的上下文 |
| `--llama3` | ✓ | ✓ | Llama3 对话模板 |
| `--zero_shot` | ✓ | ✓ | 零样本推理 |
| `--top_n` | ✓ | ✓ | 使用的段落数量（默认：10） |
| `--max_tokens` | ✓ | ✓ | 最大生成 token 数 |
| `--feedback` | ✓ | ✓ | 自反馈生成 |
| `--posthoc_at` | ✓ | ✓ | 后置引用归因 |
| `--ranking_ce` | ✓ | ✓ | 交叉编码器重排序 |
| `--reranker` | ✓ | ✓ | 重排序模型路径 |
| `--min_citation` | ✓ | ✓ | 最小引用数过滤 |
| `--norm_cite` | ✓ | ✓ | 归一化引用数 |
| `--ss_retriever` | ✓ | ✓ | Semantic Scholar 检索 |
| `--use_abstract` | ✓ | ✓ | 使用摘要进行重排序 |
| `--max_per_paper` | ✓ | ✓ | 每篇论文最大段落数 |
| `--skip_generation` | ✓ | ✓ | 跳过 LLM 生成 |
| `--sample_k` | ✓ | ✓ | 采样 K 条数据 |
| `--reverse` | ✓ | ✓ | 反转数据顺序 |
| `--start_index` | ✓ | ✓ | 从指定索引开始 |
| `--api` | ✓ | ✓* | API 提供商（忽略，使用 LM Studio） |
| `--api_key_fp` | ✓ | ✓* | API 密钥文件路径（忽略，使用 LM Studio） |
| `--download_dir` | ✓ | ✓* | 模型下载目录（忽略，使用 LM Studio） |
| `--use_slow_tokenizer` | ✓ | ✓* | 慢速分词器（忽略，使用 LM Studio） |

*这些参数为 CLI 兼容性而保留，但实际被忽略（LM Studio 负责模型加载）

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
