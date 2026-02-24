import json
import requests
from tqdm import tqdm
from sentence_transformers import CrossEncoder

API_URL = "http://localhost:1234/v1/chat/completions"

def call_lmstudio(prompt, max_tokens=1500):
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
        print(f"Error: {e}")
        return ""

def rerank_contexts(query, ctxs, reranker, top_n=10):
    if not ctxs:
        return ctxs
    pairs = [(query, f"{c.get('title', '')} {c.get('text', '')}") for c in ctxs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(ctxs, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]

def run_pipeline(input_file, output_file, reranker_model, top_n=10):
    print(f"Loading reranker: {reranker_model}")
    reranker = CrossEncoder(reranker_model)

    data = json.load(open(input_file))
    if "data" in data:
        data = data["data"]

    results = []
    for item in tqdm(data):
        query = item.get("input") or item.get("question") or item.get("query")
        ctxs = item.get("ctxs", [])

        # Rerank contexts
        reranked = rerank_contexts(query, ctxs, reranker, top_n)

        ctx_str = ""
        for idx, doc in enumerate(reranked):
            title = doc.get("title", "")
            text = doc.get("text", "")
            ctx_str += f"[{idx}] Title: {title}\nText: {text}\n\n"

        prompt = f"""Write a high-quality answer for the given question using only the provided search results. Cite results using [number] notation.

References:
{ctx_str}
Question: {query}
Answer:"""

        output = call_lmstudio(prompt)
        item["output"] = output
        item["reranked_ctxs"] = reranked
        results.append(item)
        print(f"\nQ: {query}")
        print(f"A: {output[:300]}...")

    with open(output_file, "w") as f:
        json.dump({"data": results}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="test_input.json")
    parser.add_argument("--output_file", default="test_output_reranked.json")
    parser.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()
    run_pipeline(args.input_file, args.output_file, args.reranker, args.top_n)
