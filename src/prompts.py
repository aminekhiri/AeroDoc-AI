"""AeroDoc-AI - prompts shared by ask.py, eval_answers.py and the agent (no heavy import here).

Passages are plain dicts {"file", "page", "text", "score"}, built from the retriever results
with to_passages(): the agent state and the tests do not depend on LlamaIndex objects.
"""

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

VERIFIER_SYSTEM_PROMPT = """Tu es le vérificateur d'AeroDoc. On te donne des sources numérotées, \
une question et un brouillon de réponse. Vérifie le brouillon UNIQUEMENT à partir des sources :

1. Chaque chiffre du brouillon est présent dans une source que le brouillon cite ([n]), avec la même \
valeur, la même unité, la même année et la même entité.
2. Le brouillon est complet par rapport à la question : il répond à tout ce qui est demandé.
3. Si le brouillon présente une liste comme complète (une répartition, un détail par catégorie), la somme \
des parties est égale au total annoncé.
4. Si le brouillon est un refus (« Je ne trouve pas cette information dans les documents »), il ne cite \
aucune source.

Réponds UNIQUEMENT avec un objet JSON, sans aucun texte autour :
{"verdict": "ok", "probleme": ""} si les 4 points sont respectés,
{"verdict": "a_corriger", "probleme": "<le problème précis et la correction attendue>"} sinon."""


def to_passages(nodes) -> list[dict]:
    """Convert retriever results (NodeWithScore) into plain passage dicts."""
    return [{
        "file": r.node.metadata.get("file_name", "?"),
        "page": str(r.node.metadata.get("page_label", "?")),
        "text": r.node.get_content().strip(),
        "score": float(r.score or 0.0),
    } for r in nodes]


def build_prompt(question: str, passages: list[dict]) -> str:
    """Number the passages as sources [1], [2]... and append the question."""
    blocks = [f"[{i}] {p['file']}, page {p['page']}\n{p['text']}" for i, p in enumerate(passages, 1)]
    sources = "\n\n".join(blocks)
    return f"Sources :\n\n{sources}\n\nQuestion : {question}"


def revision_prompt(question: str, passages: list[dict], previous_draft: str, feedback: str) -> str:
    """Writer prompt for a new attempt, with the verifier's comment on the previous draft."""
    return (f"{build_prompt(question, passages)}\n\n"
            f"Une vérification de ta réponse précédente a relevé ce problème :\n{feedback}\n\n"
            f"Ta réponse précédente :\n{previous_draft}\n\n"
            "Rédige une nouvelle réponse qui corrige ce problème, en respectant toutes les règles.")


def verification_prompt(question: str, passages: list[dict], draft: str) -> str:
    return f"{build_prompt(question, passages)}\n\nBrouillon de réponse à vérifier :\n{draft}"
