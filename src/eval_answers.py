"""
AeroDoc-AI - Answer evaluation: does the LLM give the right answer, cite the right page,
and refuse the questions that are not in the documents?

For each question, ONE LLM call (same retrieval and prompt as ask.py). Checks are automatic,
no second LLM is used as a judge:
  - in-corpus question: the answer contains one of the keys of data/retrieval_questions.csv
    (column answer_keys) AND cites ([n]) a source from an expected page;
  - out-of-scope question: the answer says it cannot find the information ("je ne trouve pas")
    AND cites no source (a refusal that cites sources is misleading).

Usage:
    python src/eval_answers.py                  # q01,q04,q05,q06,q09,q11,q12,q13 + 2 out-of-scope = 10 calls
    python src/eval_answers.py --dry-run        # retrieval only, no LLM call, nothing spent
    python src/eval_answers.py --ids q02,q03    # other questions (the total stays capped by --max-calls)

Answer keys: "_" = one separator (space, comma, dot), "~" = optional separator, "|" = alternatives.
Example: "22_136|22_1" matches "22,136", "22 136", "22.1".
"""
import argparse
import csv
import re
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole

from ask import SYSTEM_PROMPT, build_prompt
from config import LLM_MODEL, get_llm, get_vector_store, setup_settings
from eval_retrieval import DEFAULT_CSV, is_hit, load_questions

OUT_OF_SCOPE_CSV = Path("data/out_of_scope_questions.csv")
DEFAULT_IDS = "q01,q04,q05,q06,q09,q11,q12,q13"
SEPARATORS = "[ ,.  ]"
REFUSAL = re.compile(
    r"je ne trouve pas|je n'ai pas trouve|ne contiennent pas|"
    r"(?:i|we) (?:could not|cannot|can't|couldn't|did not|do not|don't) find|not (?:found )?in the (?:provided )?documents"
)


def key_regex(key: str) -> re.Pattern:
    """Turn an answer key such as '22_136' into a regex that tolerates number formatting."""
    parts = []
    for ch in key.strip():
        parts.append(SEPARATORS if ch == "_" else SEPARATORS + "?" if ch == "~" else re.escape(ch))
    return re.compile(r"(?<!\d)" + "".join(parts) + r"(?!\d)", re.IGNORECASE)  # words in keys are case-insensitive


def answer_has_key(answer: str, keys: str) -> bool:
    return any(key_regex(k).search(answer) for k in keys.split("|") if k.strip())


def is_refusal(answer: str) -> bool:
    plain = unicodedata.normalize("NFKD", answer.lower()).encode("ascii", "ignore").decode()
    return bool(REFUSAL.search(plain.replace("’", "'")))


def cited_sources(answer: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


def ask_llm(llm, question: str, nodes) -> str:
    response = llm.chat([
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=build_prompt(question, nodes)),
    ])
    return (response.message.content or "").strip()


def load_out_of_scope(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[error] {path} not found")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verdict_out_of_scope(refused: bool, cites_sources: bool) -> str:
    if not refused:
        return "N'A PAS REFUSE"
    return "REFUS MAIS CITE DES SOURCES" if cites_sources else "OK"


def verdict_in_corpus(retrieval_hit: bool, answer_ok: bool, cited_ok: bool, refused: bool) -> str:
    if answer_ok and cited_ok:
        return "OK"
    if refused:
        return "REFUS A TORT" if retrieval_hit else "REFUS (page non retrouvee)"
    if answer_ok:
        return "BONNE REPONSE, CITATION FAUSSE"
    return "REPONSE FAUSSE" if retrieval_hit else "REPONSE FAUSSE (page non retrouvee)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the LLM answers")
    parser.add_argument("--ids", default=DEFAULT_IDS, help="in-corpus question ids, comma separated")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-calls", type=int, default=10, help="hard cap on LLM calls for this run")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between two LLM calls")
    parser.add_argument("--dry-run", action="store_true", help="retrieval only, no LLM call")
    parser.add_argument("--out", type=Path, help="results CSV (default: data/eval/answers_<model>_<date>.csv)")
    args = parser.parse_args()

    by_id = {r["id"]: r for r in load_questions(DEFAULT_CSV)}
    wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
    unknown = [i for i in wanted if i not in by_id]
    if unknown:
        sys.exit(f"[error] unknown question ids: {', '.join(unknown)}")
    missing_keys = [i for i in wanted if not (by_id[i].get("answer_keys") or "").strip()]
    if missing_keys:
        sys.exit(f"[error] no answer_keys for: {', '.join(missing_keys)} (column answer_keys of {DEFAULT_CSV})")
    out_of_scope = load_out_of_scope(OUT_OF_SCOPE_CSV)

    total = len(wanted) + len(out_of_scope)
    if total > args.max_calls:
        sys.exit(f"[error] {total} LLM calls planned, above --max-calls={args.max_calls}. Nothing was called.")
    mode = "DRY-RUN, aucun appel LLM" if args.dry_run else f"{total} appels LLM ({LLM_MODEL}), 1 par question, plafond {args.max_calls}"
    print(f"[eval] {len(wanted)} questions du corpus + {len(out_of_scope)} hors corpus : {mode}\n", flush=True)

    setup_settings()
    retriever = VectorStoreIndex.from_vector_store(get_vector_store()).as_retriever(similarity_top_k=args.k)
    llm = None if args.dry_run else get_llm()

    jobs = [("corpus", by_id[i]) for i in wanted] + [("hors corpus", r) for r in out_of_scope]
    results, calls, consecutive_errors = [], 0, 0
    for n, (kind, row) in enumerate(jobs):
        nodes = retriever.retrieve(row["question"])
        res = {"id": row["id"], "kind": kind, "lang": row.get("lang", ""), "question": row["question"],
               "retrieval_hit": "", "answer_ok": "", "cited_ok": "", "refused": "", "verdict": "", "cited": "",
               "answer": ""}
        if kind == "corpus":
            res["retrieval_hit"] = any(is_hit(x.node.metadata, row["expected_file"], row["pages"]) for x in nodes)

        if args.dry_run:
            res["verdict"] = "DRY-RUN"
        elif consecutive_errors >= 2:
            res["verdict"] = "NON EXECUTE (2 erreurs API de suite)"
        else:
            if calls:
                time.sleep(args.pause)
            calls += 1
            try:
                answer = ask_llm(llm, row["question"], nodes)
                consecutive_errors = 0
            except Exception as exc:  # no retry of our own: the client already retries
                consecutive_errors += 1
                res["verdict"] = f"ERREUR API ({type(exc).__name__})"
                res["answer"] = str(exc)[:300]
            else:
                cites = cited_sources(answer)
                res["answer"], res["cited"] = answer, ",".join(map(str, cites))
                refused = is_refusal(answer)
                if kind == "corpus":
                    res["answer_ok"] = answer_has_key(answer, row["answer_keys"]) and not refused
                    res["cited_ok"] = any(
                        is_hit(nodes[c - 1].node.metadata, row["expected_file"], row["pages"])
                        for c in cites if 1 <= c <= len(nodes))
                    res["verdict"] = verdict_in_corpus(res["retrieval_hit"], res["answer_ok"], res["cited_ok"], refused)
                else:
                    res["verdict"] = verdict_out_of_scope(refused, bool(cites))
                res["refused"] = refused
                if refused and cites and kind == "corpus":
                    res["verdict"] += " + CITATIONS"
        results.append(res)

        shown = " ".join(res["answer"].split())[:95]
        print(f"{res['id']:<5}{kind:<12}{res['verdict']:<38}{shown}", flush=True)

    done = [r for r in results if r["verdict"] and not r["verdict"].startswith(("ERREUR", "NON EXEC", "DRY"))]
    corpus = [r for r in done if r["kind"] == "corpus"]
    oos = [r for r in done if r["kind"] == "hors corpus"]
    print(f"\n=== Appels LLM effectues : {calls} (plafond {args.max_calls}) ===")
    if args.dry_run:
        hits = sum(bool(r["retrieval_hit"]) for r in results if r["kind"] == "corpus")
        print(f"Recherche seule : bonne page dans le top {args.k} pour {hits}/{len(wanted)} questions du corpus.")
        return
    if corpus:
        print(f"Questions du corpus : bonne reponse {sum(bool(r['answer_ok']) for r in corpus)}/{len(corpus)}, "
              f"bonne citation {sum(bool(r['cited_ok']) for r in corpus)}/{len(corpus)}, "
              f"les deux (OK) {sum(r['verdict'] == 'OK' for r in corpus)}/{len(corpus)}")
    if oos:
        print(f"Hors corpus : refus correct (sans citation) {sum(r['verdict'] == 'OK' for r in oos)}/{len(oos)}")
    refusals = [r for r in done if r["refused"] is True]
    if refusals:
        print(f"Refus qui citent des sources : {sum(bool(r['cited']) for r in refusals)}/{len(refusals)}")
    errors = [r for r in results if r["verdict"].startswith(("ERREUR", "NON EXEC"))]
    if errors:
        print(f"Non evaluees (erreur API ou arret) : {', '.join(r['id'] for r in errors)}")

    out = args.out or Path(f"data/eval/answers_{LLM_MODEL}_{date.today().isoformat()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"Resultats (reponses completes) : {out}")


if __name__ == "__main__":
    main()
