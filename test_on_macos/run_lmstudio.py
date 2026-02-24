import json
import requests
from tqdm import tqdm

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

def run_openscholar(input_file, output_file, top_n=10):
    data = json.load(open(input_file))
    if "data" in data:
        data = data["data"]

    results = []
    for item in tqdm(data):
        query = item.get("input") or item.get("question") or item.get("query")
        ctxs = item.get("ctxs", [])

        ctx_str = ""
        for idx, doc in enumerate(ctxs[:top_n]):
            title = doc.get("title", "")
            text = doc.get("text", "")
            if title:
                ctx_str += f"[{idx}] Title: {title} Text: {text}\n"
            else:
                ctx_str += f"[{idx}] {text}\n"

        prompt = f"""Write a high-quality answer for the given question using only the provided search results. Cite results using [number] notation.

References:
{ctx_str}
Question: {query}
Answer:"""

        output = call_lmstudio(prompt)
        item["output"] = output
        results.append(item)
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print(f"A: {output[:500]}..." if len(output) > 500 else f"A: {output}")

    with open(output_file, "w") as f:
        json.dump({"data": results}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="test_input.json")
    parser.add_argument("--output_file", default="test_output.json")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()
    run_openscholar(args.input_file, args.output_file, args.top_n)
