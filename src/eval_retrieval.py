"""
AeroDoc-AI - Retrieval evaluation: recall@k on a CSV of questions (no LLM).

For each question, the retriever returns the top-k chunks. The question is a HIT
if at least one of them comes from the expected file AND from one of the
expected pages. Recall@k = hits / number of questions.

Usage:
    python src/eval_retrieval.py                      # recall@5 on data/retrieval_questions.csv
    python src/eval_retrieval.py --k 10 --verbose     # show the retrieved chunks of each miss
    python src/eval_retrieval.py --out data/eval/baseline.csv

CSV columns used: id, lang, question, expected_file, expected_page_labels
(pages separated by "|"). Pages are the PRINTED page labels, which is what
LlamaIndex stores as "page_label" (not the physical PDF page index).
"""
import argparse
import csv
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

from llama_index.core import VectorStoreIndex

from config import EMBED_MODEL, get_vector_store, setup_settings

DEFAULT_CSV = Path("data/retrieval_questions.csv")


def normalize_filename(name: str) -> str:
    """Make file names comparable: case-insensitive, spaces and underscores equivalent."""
    return re.sub(r"[\s_]+", "_", name.strip()).lower()


def load_questions(path: Path) -> list[dict]:
    """Read the evaluation CSV and split expected_page_labels into a set of labels."""
    if not path.exists():
        sys.exit(f"[error] {path} not found")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"id", "question", "expected_file", "expected_page_labels"}
    if not rows or not required <= set(rows[0]):
        sys.exit(f"[error] {path} must contain the columns: {', '.join(sorted(required))}")
    for row in rows:
        row["pages"] = {p.strip() for p in row["expected_page_labels"].split("|") if p.strip()}
    return rows


def is_hit(node_metadata: dict, expected_file: str, expected_pages: set[str]) -> bool:
    """True if a retrieved chunk comes from the expected file and one of the expected pages."""
    same_file = normalize_filename(node_metadata.get("file_name", "")) == normalize_filename(expected_file)
    same_page = str(node_metadata.get("page_label", "")).strip() in expected_pages
    return same_file and same_page


def evaluate(rows: list[dict], k: int, verbose: bool) -> list[dict]:
    """Run every question through the retriever and record the rank of the first hit."""
    index = VectorStoreIndex.from_vector_store(get_vector_store())
    retriever = index.as_retriever(similarity_top_k=k)

    results = []
    for row in rows:
        retrieved = retriever.retrieve(row["question"])
        rank = next(
            (i for i, r in enumerate(retrieved, 1)
             if is_hit(r.node.metadata, row["expected_file"], row["pages"])),
            None,
        )
        results.append({
            "id": row["id"],
            "lang": row.get("lang", ""),
            "file": row["expected_file"],
            "hit": rank is not None,
            "rank": rank or "",
            "expected_pages": "|".join(sorted(row["pages"], key=lambda p: (len(p), p))),
            "retrieved": " ; ".join(
                f"{r.node.metadata.get('file_name', '?')} p.{r.node.metadata.get('page_label', '?')}"
                for r in retrieved
            ),
            "question": row["question"],
        })

        status = f"HIT  (rank {rank})" if rank else "MISS"
        print(f"{row['id']:<5}{status:<14}{row['question'][:75]}")
        if verbose and not rank:
            print(f"      expected: {row['expected_file']} p.{'|'.join(sorted(row['pages']))}")
            for i, r in enumerate(retrieved, 1):
                m = r.node.metadata
                snippet = textwrap.shorten(r.node.get_content().replace("\n", " "), width=110)
                print(f"      [{i}] {m.get('file_name', '?')} p.{m.get('page_label', '?')} "
                      f"(score {r.score:.3f}) {snippet}")
    return results


def summarize(results: list[dict], k: int) -> None:
    """Print recall@k overall, by language and by document, plus MRR."""
    def recall(items: list[dict]) -> str:
        hits = sum(r["hit"] for r in items)
        return f"{hits}/{len(items)} = {hits / len(items):.1%}"

    print(f"\n=== Recall@{k} : {recall(results)} ===")
    mrr = sum(1 / r["rank"] for r in results if r["hit"]) / len(results)
    print(f"MRR : {mrr:.3f}  (1.0 = bon passage toujours en 1re position)")

    for title, key in (("Par langue de la question", "lang"), ("Par document", "file")):
        groups = defaultdict(list)
        for r in results:
            groups[r[key] or "?"].append(r)
        print(f"\n{title} :")
        for name, items in sorted(groups.items()):
            print(f"  {name:<52}{recall(items)}")

    misses = [r["id"] for r in results if not r["hit"]]
    if misses:
        print(f"\nQuestions manquées : {', '.join(misses)}")


def write_csv(results: list[dict], path: Path) -> None:
    """Save the per-question results so two runs can be compared."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nRésultats enregistrés dans {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@k of the retriever")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="questions file")
    parser.add_argument("--k", type=int, default=5, help="number of chunks retrieved per question")
    parser.add_argument("--verbose", action="store_true", help="show the retrieved chunks of each miss")
    parser.add_argument("--out", type=Path, help="write per-question results to this CSV")
    args = parser.parse_args()

    rows = load_questions(args.csv)
    setup_settings()
    print(f"[eval] {len(rows)} questions, k={args.k}, model={EMBED_MODEL}\n")

    results = evaluate(rows, args.k, args.verbose)
    summarize(results, args.k)
    if args.out:
        write_csv(results, args.out)


if __name__ == "__main__":
    main()
