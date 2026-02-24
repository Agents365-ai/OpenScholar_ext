import json
import argparse
import re
from pathlib import Path

def generate_pdf(md_path, pdf_path):
    """Convert Markdown to PDF using pandoc or md2pdf"""
    import subprocess

    # Try pandoc first (most reliable)
    try:
        subprocess.run([
            "pandoc", md_path,
            "-o", pdf_path,
            "--pdf-engine=xelatex",
            "-V", "geometry:margin=1in",
            "-V", "mainfont=Arial"
        ], check=True, capture_output=True)
        print(f"PDF saved to {pdf_path}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try md2pdf
    try:
        from md2pdf.core import md2pdf as convert_md2pdf
        convert_md2pdf(pdf_path, md_content=Path(md_path).read_text())
        print(f"PDF saved to {pdf_path}")
        return True
    except ImportError:
        pass

    print("Warning: Install pandoc or md2pdf for PDF generation")
    print("  brew install pandoc  OR  pip install md2pdf")
    return False

def extract_citations(text):
    """Extract citation numbers from text like [0], [1], etc."""
    return list(set(re.findall(r'\[(\d+)\]', text)))

def generate_markdown(data, output_path):
    md = "# OpenScholar Results\n\n"

    for i, item in enumerate(data, 1):
        query = item.get("input") or item.get("question") or item.get("query", "")
        output = item.get("output", "")
        ctxs = item.get("ctxs", [])

        md += f"## Question {i}\n\n"
        md += f"**{query}**\n\n"
        md += f"### Answer\n\n{output}\n\n"

        # Extract cited references
        cited = extract_citations(output)
        if ctxs:
            md += "### References\n\n"
            for idx, ctx in enumerate(ctxs):
                title = ctx.get("title", "Untitled")
                url = ctx.get("url", "")
                citations = ctx.get("citation_counts", 0)
                abstract = ctx.get("abstract", "") or ctx.get("text", "")

                cited_marker = " ✓" if str(idx) in cited else ""
                if url:
                    md += f"**[{idx}]** [{title}]({url}){cited_marker}\n"
                else:
                    md += f"**[{idx}]** {title}{cited_marker}\n"
                if citations:
                    md += f"   - Citations: {citations}\n"
                if abstract:
                    md += f"   - {abstract[:200]}{'...' if len(abstract) > 200 else ''}\n"
                md += "\n"
        md += "---\n\n"

    Path(output_path).write_text(md, encoding="utf-8")
    print(f"Markdown saved to {output_path}")

def generate_html(data, output_path):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenScholar Results</title>
    <style>
        :root {
            --primary: #2563eb;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #1e293b;
            --muted: #64748b;
            --border: #e2e8f0;
            --accent: #dbeafe;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: var(--primary);
            margin-bottom: 2rem;
            font-size: 2rem;
        }
        .qa-card {
            background: var(--card);
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            overflow: hidden;
        }
        .question {
            background: var(--primary);
            color: white;
            padding: 1rem 1.5rem;
            font-weight: 600;
            font-size: 1.1rem;
        }
        .question-num {
            background: rgba(255,255,255,0.2);
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            margin-right: 0.5rem;
            font-size: 0.85rem;
        }
        .answer {
            padding: 1.5rem;
            background: var(--accent);
            border-bottom: 1px solid var(--border);
        }
        .answer-label {
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .answer-text {
            white-space: pre-wrap;
        }
        .citation {
            background: var(--primary);
            color: white;
            padding: 0.1rem 0.4rem;
            border-radius: 3px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
        }
        .citation:hover { opacity: 0.8; }
        .references {
            padding: 1.5rem;
        }
        .references-label {
            font-weight: 600;
            color: var(--muted);
            margin-bottom: 1rem;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .ref-item {
            padding: 0.75rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            border: 1px solid var(--border);
            transition: all 0.2s;
        }
        .ref-item:hover { border-color: var(--primary); }
        .ref-item.cited { background: #f0fdf4; border-color: #22c55e; }
        .ref-num {
            display: inline-block;
            background: var(--muted);
            color: white;
            padding: 0.1rem 0.5rem;
            border-radius: 3px;
            font-size: 0.8rem;
            margin-right: 0.5rem;
        }
        .ref-item.cited .ref-num { background: #22c55e; }
        .ref-title {
            font-weight: 600;
            color: var(--text);
        }
        .ref-title a { color: var(--primary); text-decoration: none; }
        .ref-title a:hover { text-decoration: underline; }
        .ref-meta {
            font-size: 0.85rem;
            color: var(--muted);
            margin-top: 0.25rem;
        }
        .ref-abstract {
            font-size: 0.85rem;
            color: var(--muted);
            margin-top: 0.5rem;
            padding-left: 1rem;
            border-left: 2px solid var(--border);
        }
        .badge {
            display: inline-block;
            background: var(--border);
            padding: 0.1rem 0.5rem;
            border-radius: 10px;
            font-size: 0.75rem;
            margin-left: 0.5rem;
        }
        .badge.cited { background: #dcfce7; color: #166534; }
        footer {
            text-align: center;
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 2rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 OpenScholar Results</h1>
"""

    for i, item in enumerate(data, 1):
        query = item.get("input") or item.get("question") or item.get("query", "")
        output = item.get("output", "")
        ctxs = item.get("ctxs", [])
        cited = extract_citations(output)

        # Highlight citations in answer
        answer_html = output
        for c in sorted(set(re.findall(r'\[\d+\]', output)), key=lambda x: -int(x[1:-1])):
            answer_html = answer_html.replace(c, f'<span class="citation">{c}</span>')
        answer_html = answer_html.replace('\n', '<br>')

        html += f"""
        <div class="qa-card">
            <div class="question">
                <span class="question-num">Q{i}</span> {query}
            </div>
            <div class="answer">
                <div class="answer-label">Answer</div>
                <div class="answer-text">{answer_html}</div>
            </div>
            <div class="references">
                <div class="references-label">References ({len(cited)} cited / {len(ctxs)} total)</div>
"""

        for idx, ctx in enumerate(ctxs):
            title = ctx.get("title", "Untitled")
            url = ctx.get("url", "")
            citations = ctx.get("citation_counts", 0)
            abstract = ctx.get("abstract", "") or ctx.get("text", "")
            is_cited = str(idx) in cited

            cited_class = " cited" if is_cited else ""
            cited_badge = '<span class="badge cited">Cited</span>' if is_cited else ""

            title_html = f'<a href="{url}" target="_blank">{title}</a>' if url else title
            citations_html = f'<span class="badge">📊 {citations:,} citations</span>' if citations else ""

            html += f"""
                <div class="ref-item{cited_class}">
                    <div class="ref-title">
                        <span class="ref-num">{idx}</span>
                        {title_html}
                        {cited_badge}
                        {citations_html}
                    </div>
"""
            if abstract:
                abstract_short = abstract[:250] + "..." if len(abstract) > 250 else abstract
                html += f'<div class="ref-abstract">{abstract_short}</div>'

            html += "</div>"

        html += """
            </div>
        </div>
"""

    html += """
        <footer>
            Generated by OpenScholar • <a href="https://github.com/AkariAsai/OpenScholar">GitHub</a>
        </footer>
    </div>
</body>
</html>
"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"HTML saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize OpenScholar results")
    parser.add_argument("--input", "-i", default="test_output.json", help="Input JSON file")
    parser.add_argument("--output", "-o", default="results", help="Output filename (without extension)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if "data" in data:
        data = data["data"]

    generate_markdown(data, f"{args.output}.md")
    generate_html(data, f"{args.output}.html")
    generate_pdf(f"{args.output}.md", f"{args.output}.pdf")
    print(f"\nOpen {args.output}.html in browser or {args.output}.pdf")

if __name__ == "__main__":
    main()
