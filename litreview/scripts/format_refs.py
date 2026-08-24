"""Turn the final top-K candidates into a compact, numbered reference brief
(title/authors/year/venue/abstract) for the generation step -- this is the
context that would normally be fed to the generator LLM's prompt in QUAL-SG's
RAG-based task. Since no hosted LLM API key is available in this environment,
Claude performs the outline + section-writing generation itself, using this
file as the retrieved-context input (playing the same role the prompt would)."""
from common import load_json


def main():
    D = load_json("../data/candidates_topK.json")
    lines = []
    for i, w in enumerate(D, 1):
        authors = w.get("authors") or []
        author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        lines.append(
            f"[{i}] {w['title']} ({w.get('year')}). {author_str}. "
            f"Venue: {w.get('venue') or 'arXiv'}. Citations: {w.get('cited_by_count', 0)}. "
            f"Source: {w.get('source','openalex')}.\n"
            f"    Abstract: {(w.get('abstract') or '')[:500]}\n"
        )
    with open("../data/reference_brief.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(D)} references to ../data/reference_brief.txt")


if __name__ == "__main__":
    main()
