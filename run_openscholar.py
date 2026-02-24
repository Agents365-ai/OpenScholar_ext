"""
OpenScholar macOS Pipeline - Full compatibility with run.py
Supports 3 modes:
1. Standard RAG: retrieve + generate
2. Reranker Pipeline: retrieve + rerank + generate
3. Self-reflective: retrieve + rerank + feedback + generate
"""
import os
import json
import requests
import time
import argparse
import random
from tqdm import tqdm

S2_API_KEY = os.getenv("S2_API_KEY", "")
API_URL = os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")

# ============ Data Loading ============
def load_jsonlines(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_input_data(args):
    """Load input data from file or HuggingFace dataset"""
    if args.input_file:
        if args.input_file.endswith(".jsonl"):
            data = load_jsonlines(args.input_file)
        else:
            data = json.load(open(args.input_file))
            if "data" in data:
                data = data["data"]
    elif args.dataset:
        import datasets
        data = list(datasets.load_dataset(args.dataset)["test"])
    else:
        return None

    # Sample if requested
    if args.sample_k > 0:
        random.shuffle(data)
        data = data[:args.sample_k]

    # Start from index
    if args.start_index:
        data = data[args.start_index:]

    # Reverse order
    if args.reverse:
        data = data[::-1]

    return data

# ============ Retrieval ============
def search_s2(query, limit=20):
    """Search Semantic Scholar API"""
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,url,citationCount,year,authors"
    }
    resp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers=headers, params=params, timeout=30
    )
    if resp.status_code != 200:
        print(f"S2 API error: {resp.status_code}")
        return []
    return resp.json().get("data", [])

def format_ctxs(papers):
    """Format papers to ctxs format"""
    ctxs = []
    for p in papers:
        if not p.get("abstract"):
            continue
        authors = ", ".join([a["name"] for a in p.get("authors", [])[:3]])
        ctxs.append({
            "title": p.get("title", ""),
            "text": p.get("abstract", ""),
            "abstract": p.get("abstract", ""),
            "url": p.get("url", ""),
            "citation_counts": p.get("citationCount", 0),
            "year": p.get("year"),
            "authors": authors
        })
    return ctxs

# ============ Reranking ============
def load_reranker(model_name):
    """Load reranker model"""
    try:
        from FlagEmbedding import FlagReranker
        return FlagReranker(model_name, use_fp16=True)
    except ImportError:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)

def rerank_ctxs(query, ctxs, reranker, top_n=10, use_abstract=False):
    """Rerank contexts by relevance"""
    if not ctxs or reranker is None:
        return ctxs[:top_n]

    pairs = []
    for c in ctxs:
        text = c.get("abstract", "") if use_abstract else c.get("text", "")
        pairs.append([query, f"{c.get('title', '')} {text}"])

    if hasattr(reranker, 'compute_score'):
        scores = reranker.compute_score(pairs)
        if isinstance(scores, float):
            scores = [scores]
    else:
        scores = reranker.predict(pairs)

    ranked = sorted(zip(ctxs, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]

def filter_by_citation(ctxs, min_citation):
    """Filter contexts by minimum citation count"""
    if not min_citation:
        return ctxs
    return [c for c in ctxs if c.get("citation_counts", 0) >= min_citation]

def normalize_citations(ctxs):
    """Normalize citation counts for ranking"""
    if not ctxs:
        return ctxs
    max_cite = max(c.get("citation_counts", 0) for c in ctxs) or 1
    for c in ctxs:
        c["norm_citation"] = c.get("citation_counts", 0) / max_cite
    return ctxs

def limit_per_paper(ctxs, max_per_paper=3):
    """Limit passages per paper"""
    if not max_per_paper:
        return ctxs
    seen = {}
    result = []
    for c in ctxs:
        title = c.get("title", "")
        if title not in seen:
            seen[title] = 0
        if seen[title] < max_per_paper:
            result.append(c)
            seen[title] += 1
    return result

# ============ Generation ============
def call_llm(prompt, max_tokens=1500):
    """Call LM Studio API"""
    payload = {
        "model": "llama-3.1_openscholar-8b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM API error: {e}")
        return ""

def build_prompt(query, ctxs, use_abstract=False, task_name="default"):
    """Build RAG prompt with references"""
    ctx_str = ""
    for idx, doc in enumerate(ctxs):
        title = doc.get("title", "")
        text = doc.get("abstract", "") if use_abstract else doc.get("text", "")
        ctx_str += f"[{idx}] Title: {title}\nText: {text}\n\n"

    # Task-specific prompts
    if task_name == "scifact":
        return f"""Based on the provided references, determine if the following claim is supported, refuted, or has not enough info. Cite references using [number].

References:
{ctx_str}
Claim: {query}
Verdict:"""
    elif task_name == "pubmedqa":
        return f"""Based on the provided references, answer the following yes/no/maybe question. Cite references using [number].

References:
{ctx_str}
Question: {query}
Answer:"""
    elif task_name == "qasa":
        return f"""Based on the provided references, provide a detailed answer. Cite references using [number].

References:
{ctx_str}
Question: {query}
Answer:"""
    else:
        return f"""Write a high-quality answer for the given question using only the provided search results. Cite results using [number] notation.

References:
{ctx_str}
Question: {query}
Answer:"""

def generate_with_feedback(query, ctxs, use_abstract=False, task_name="default", max_tokens=1500):
    """Self-reflective generation with feedback loop"""
    prompt = build_prompt(query, ctxs, use_abstract, task_name)
    output = call_llm(prompt, max_tokens)

    feedback_prompt = f"""Review this answer and identify any missing information or inaccuracies based on the references.

Question: {query}
Answer: {output}

What additional information from the references should be included? Be specific."""

    feedback = call_llm(feedback_prompt, max_tokens=500)

    refined_prompt = f"""{prompt}

Previous answer: {output}
Feedback: {feedback}

Write an improved answer incorporating the feedback:"""

    refined_output = call_llm(refined_prompt, max_tokens)
    return refined_output if refined_output else output

# ============ Pipeline ============
def process_item(item, args):
    """Process a single item through the pipeline"""
    query = item.get("input") or item.get("question") or item.get("query")
    ctxs = item.get("ctxs", [])

    # Normalize ctx format
    for ctx in ctxs:
        if "retrieval text" in ctx:
            ctx["text"] = ctx["retrieval text"]
        if ctx.get("text") is None:
            ctx["text"] = ""

    # Step 1: Retrieve if needed
    if args.ss_retriever and not ctxs:
        print(f"Retrieving papers for: {query[:50]}...")
        papers = search_s2(query, limit=args.top_n * 3)
        ctxs = format_ctxs(papers)
        time.sleep(0.3)

    # Step 2: Filter by minimum citation
    if args.min_citation:
        ctxs = filter_by_citation(ctxs, args.min_citation)

    # Step 3: Normalize citations
    if args.norm_cite:
        ctxs = normalize_citations(ctxs)

    # Step 4: Rerank
    if args.ranking_ce and args.reranker:
        if not hasattr(args, '_reranker_model'):
            print(f"Loading reranker: {args.reranker}")
            args._reranker_model = load_reranker(args.reranker)
        ctxs = rerank_ctxs(query, ctxs, args._reranker_model, args.top_n, args.use_abstract)
    else:
        ctxs = ctxs[:args.top_n]

    # Step 5: Limit per paper
    if args.max_per_paper:
        ctxs = limit_per_paper(ctxs, args.max_per_paper)

    # Step 6: Generate (skip if requested)
    if args.skip_generation:
        output = ""
    elif args.feedback:
        output = generate_with_feedback(query, ctxs, args.use_abstract, args.task_name, args.max_tokens)
    else:
        prompt = build_prompt(query, ctxs, args.use_abstract, args.task_name)
        output = call_llm(prompt, args.max_tokens)

    return {
        "input": query,
        "ctxs": ctxs,
        "output": output,
        "answer": item.get("answer", "")
    }

def main():
    parser = argparse.ArgumentParser(description="OpenScholar macOS Pipeline (run.py compatible)")

    # Input/Output
    parser.add_argument("--input_file", "-i", type=str, help="Path to input file")
    parser.add_argument("--output_file", "-o", type=str, default="output.json", help="Path to output file")
    parser.add_argument("--query", "-q", type=str, help="Single question (interactive mode)")

    # Data loading
    parser.add_argument("--task_name", type=str, default="default",
                       help="Task: default, scifact, pubmedqa, qasa")
    parser.add_argument("--dataset", type=str, default=None,
                       help="HuggingFace dataset path")

    # Model config (for compatibility, uses LM Studio)
    parser.add_argument("--model_name", type=str, default="OpenScholar/Llama-3.1_OpenScholar-8B",
                       help="Model name (display only, uses LM Studio)")
    parser.add_argument("--use_contexts", action="store_true",
                       help="Use contexts from input file")
    parser.add_argument("--llama3", action="store_true", help="Use Llama3 chat template")
    parser.add_argument("--zero_shot", action="store_true", help="Zero-shot inference")

    # API config (for run.py compatibility, not used in LM Studio mode)
    parser.add_argument("--api", type=str, default=None,
                       help="API provider (ignored, uses LM Studio)")
    parser.add_argument("--api_key_fp", type=str, default=None,
                       help="API key file path (ignored, uses LM Studio)")
    parser.add_argument("--download_dir", type=str, default="./cache",
                       help="Model download dir (ignored, uses LM Studio)")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                       help="Use slow tokenizer (ignored, uses LM Studio)")

    # Generation config
    parser.add_argument("--top_n", type=int, default=10,
                       help="Number of passages to use")
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--feedback", action="store_true",
                       help="Enable self-reflective feedback")
    parser.add_argument("--posthoc_at", action="store_true",
                       help="Post-hoc attribution")

    # Reranking config
    parser.add_argument("--ranking_ce", action="store_true",
                       help="Enable cross-encoder reranking")
    parser.add_argument("--reranker", type=str, default="OpenScholar/OpenScholar_Reranker",
                       help="Reranker model name")
    parser.add_argument("--min_citation", type=int, default=None,
                       help="Minimum citation count filter")
    parser.add_argument("--norm_cite", action="store_true",
                       help="Normalize citation counts")
    parser.add_argument("--ss_retriever", action="store_true",
                       help="Use Semantic Scholar retrieval")
    parser.add_argument("--use_abstract", action="store_true",
                       help="Use abstract for reranking")
    parser.add_argument("--max_per_paper", type=int, default=None,
                       help="Max passages per paper")

    # Debug options
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip generation (for debugging)")
    parser.add_argument("--sample_k", type=int, default=-1,
                       help="Sample K items from data")
    parser.add_argument("--reverse", action="store_true",
                       help="Reverse data order")
    parser.add_argument("--start_index", type=int, default=None,
                       help="Start from index")

    args = parser.parse_args()

    # Single query mode
    if args.query:
        if not args.use_contexts:
            args.ss_retriever = True
        item = {"input": args.query, "ctxs": []}
        result = process_item(item, args)
        print(f"\nQ: {result['input']}")
        print(f"A: {result['output']}")
        with open(args.output_file, "w") as f:
            json.dump({"data": [result]}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output_file}")
        return

    # Batch mode
    data = load_input_data(args)
    if data is None:
        parser.print_help()
        print("\n" + "="*60)
        print("EXAMPLES (compatible with run.py):")
        print("="*60)
        print("\n# 1. Standard RAG Pipeline")
        print("python run_openscholar.py \\")
        print("    --input_file YOUR_INPUT_FILE \\")
        print("    --use_contexts \\")
        print("    --output_file output.json \\")
        print("    --top_n 10 --llama3 --zero_shot")
        print("\n# 2. Retriever + Reranker Pipeline")
        print("python run_openscholar.py \\")
        print("    --input_file YOUR_INPUT_FILE \\")
        print("    --use_contexts --ranking_ce \\")
        print("    --reranker OpenScholar/OpenScholar_Reranker \\")
        print("    --output_file output.json \\")
        print("    --top_n 10 --llama3 --zero_shot")
        print("\n# 3. Self-reflective Generation Pipeline")
        print("python run_openscholar.py \\")
        print("    --input_file YOUR_INPUT_FILE \\")
        print("    --use_contexts --ranking_ce \\")
        print("    --reranker OpenScholar/OpenScholar_Reranker \\")
        print("    --feedback --use_abstract --norm_cite \\")
        print("    --max_per_paper 3 \\")
        print("    --output_file output.json \\")
        print("    --top_n 10 --llama3 --zero_shot")
        print("\n# 4. With S2 Retrieval + Task-specific")
        print("python run_openscholar.py \\")
        print("    -q \"What causes Alzheimer's disease?\" \\")
        print("    --ss_retriever --task_name pubmedqa")
        return

    # Resume from existing output
    final_results = []
    if os.path.isfile(args.output_file):
        existing = json.load(open(args.output_file))
        final_results = existing.get("data", [])
        data = data[len(final_results):]
        print(f"Resuming from {len(final_results)} existing results")

    # Process data
    for idx, item in enumerate(tqdm(data)):
        start = time.time()
        result = process_item(item, args)
        result["elapsed"] = time.time() - start
        final_results.append(result)

        # Save periodically
        if idx % 10 == 0:
            with open(args.output_file, "w") as f:
                json.dump({"data": final_results}, f, indent=2, ensure_ascii=False)

        print(f"\nQ: {result['input'][:80]}...")
        print(f"A: {result['output'][:200]}...")

    # Final save
    with open(args.output_file, "w") as f:
        json.dump({"data": final_results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(final_results)} results to {args.output_file}")

if __name__ == "__main__":
    main()
