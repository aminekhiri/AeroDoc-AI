"""
AeroDoc-AI - Ask a question: retrieval (pgvector) + answer written by Gemini, with citations.

Usage:
    python src/ask.py "What are the requirements for emergency exits?"
    python src/ask.py "Combien d'avions Airbus a-t-il livrés en 2023 ?" --k 5
    python src/ask.py "..." --show-prompt     # print the prompt, do not call the LLM (no API key needed)

Needs GEMINI_API_KEY in .env (see .env.example).
"""
import argparse
import re
import sys

print("[ask] chargement des bibliothèques (10 à 30 s la première fois)...", flush=True)

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole

from config import LLM_MODEL, get_llm, get_vector_store, setup_settings

SYSTEM_PROMPT = """Tu es AeroDoc, un assistant qui répond à des questions sur des documents publics \
du secteur aéronautique (rapports annuels, spécifications de certification EASA).

Règles :
- Réponds UNIQUEMENT à partir des sources fournies. N'utilise aucune connaissance extérieure.
- Cite la source de chaque information avec son numéro entre crochets, par exemple [1] ou [2][3].
- Reprends les chiffres exactement comme dans la source, avec leur unité et leur année. Vérifie que \
l'entreprise, l'année et l'indicateur de la source correspondent bien à la question.
- Si les sources ne contiennent pas l'information demandée, réponds uniquement : \
"Je ne trouve pas cette information dans les documents." Dans ce cas, n'ajoute aucune citation [n] \
et aucune explication. N'invente rien et ne devine pas.
- Réponds dans la langue de la question, de façon concise."""


def build_prompt(question: str, nodes) -> str:
    """Number the retrieved chunks as sources [1], [2]... and append the question."""
    blocks = []
    for i, r in enumerate(nodes, 1):
        meta = r.node.metadata
        header = f"[{i}] {meta.get('file_name', '?')}, page {meta.get('page_label', '?')}"
        blocks.append(f"{header}\n{r.node.get_content().strip()}")
    sources = "\n\n".join(blocks)
    return f"Sources :\n\n{sources}\n\nQuestion : {question}"


def error_hint(exc: Exception) -> str:
    """Short advice for the most common Gemini API errors."""
    text = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429 or "429" in text or "resource_exhausted" in text or "quota" in text:
        return ("\n[hint] Limite de débit ou de quota atteinte sur votre compte Gemini : "
                "attendez une minute, vérifiez vos quotas sur aistudio.google.com, "
                "ou essayez un autre modèle (LLM_MODEL dans .env).")
    if code == 404 or "not found" in text:
        return "\n[hint] Modèle introuvable : vérifiez LLM_MODEL dans .env."
    if "api key" in text or code in (401, 403):
        return "\n[hint] Clé refusée : vérifiez GEMINI_API_KEY dans .env."
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5, help="number of passages given to the LLM")
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt and stop")
    args = parser.parse_args()

    setup_settings()
    index = VectorStoreIndex.from_vector_store(get_vector_store())
    nodes = index.as_retriever(similarity_top_k=args.k).retrieve(args.question)
    prompt = build_prompt(args.question, nodes)

    if args.show_prompt:
        print(f"\n--- system ---\n{SYSTEM_PROMPT}\n\n--- user ---\n{prompt}")
        return

    print(f"[ask] {len(nodes)} passages trouvés, appel à Gemini ({LLM_MODEL})...", flush=True)
    try:
        llm = get_llm()
        response = llm.chat([
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ])
    except RuntimeError as exc:  # missing key
        sys.exit(f"[error] {exc}")
    except Exception as exc:  # network, quota, invalid key... (the client already retries 3 times)
        sys.exit(f"[error] Gemini call failed ({LLM_MODEL}): {type(exc).__name__}: {exc}{error_hint(exc)}")

    answer = (response.message.content or "").strip()
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}

    print(f"\nQ: {args.question}\n\n{answer}\n\nSources ({LLM_MODEL}) :")
    for i, r in enumerate(nodes, 1):
        meta = r.node.metadata
        mark = "*" if i in cited else " "
        print(f" {mark}[{i}] {meta.get('file_name', '?')} p.{meta.get('page_label', '?')}  (score {r.score:.3f})")
    print("(* = source citée dans la réponse)")


if __name__ == "__main__":
    main()
