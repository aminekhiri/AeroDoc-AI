# AeroDoc-AI

AeroDoc-AI est un assistant intelligent (RAG & Text-to-SQL) conçu pour interroger et analyser des documents publics du secteur aéronautique (rapports annuels d'Airbus, Thales, Safran, spécifications EASA).

## 🚀 Fonctionnalités principales

*   **RAG avec citations fiables** : Recherche dans les documents et réponses sourcées (nom du document et page). L'assistant indique clairement s'il ne possède pas l'information.
*   **Analyse de données structurées** : Extraction des chiffres clés (effectifs, chiffre d'affaires, R&D) et interrogation via Text-to-SQL.
*   **Architecture multi-agents (LangGraph)** :
    *   *Routeur* : Oriente la question vers la recherche textuelle (RAG) ou la base de données structurée (SQL).
    *   *Rédacteur* : Formule la réponse à partir des informations récupérées.
    *   *Vérificateur* : Contrôle l'exactitude des chiffres et des citations avant de valider la réponse finale (avec un système de correction itérative).

## 🛠️ Stack Technique

*   **Stockage vectoriel** : pgvector
*   **Base de données analytique** : DuckDB
*   **Orchestration d'agents** : LangGraph
*   **Évaluation & Suivi** : MLflow
*   **API & Déploiement** : FastAPI, Docker, Google Cloud Run
*   **Interface** : Streamlit (Optionnel)

## 📁 Structure du projet

```
AeroDoc-AI/
├── src/
│   ├── config.py        # configuration partagée (modèle d'embedding, pgvector)
│   ├── ingest.py        # PDF -> passages -> embeddings -> pgvector
│   └── search.py        # test de recherche (top-k passages d'une question)
├── scripts/
│   └── download_corpus.py   # télécharge les PDF et écrit data/manifest.csv
├── data/
│   ├── raw/             # PDF du corpus (non suivis par Git)
│   └── manifest.csv     # liste des documents du corpus
├── docs/                # roadmap et documentation
├── docker-compose.yml   # base Postgres + pgvector
├── requirements.txt
└── .env.example         # variables d'environnement (copier vers .env)
```

Les commandes se lancent depuis la racine du projet :

```
python scripts/download_corpus.py
python src/ingest.py --reset
python src/search.py "votre question" --k 5
```

## 🗺️ Roadmap du Projet

Ce projet est conçu pour être réalisé en itérations courtes :

1.  **Cadrage** : Structure du projet et constitution du corpus (5 à 8 documents).
2.  **Ingestion & RAG** : Extraction du texte des PDF, découpage, embeddings, et mise en place de la recherche avec citations strictes.
3.  **SQL & Agents** : Extraction des tableaux vers DuckDB, outil Text-to-SQL, et mise en place du workflow LangGraph (Routeur -> Rédacteur -> Vérificateur).
4.  **Évaluation & Déploiement** : Benchmark sur 30-40 questions pour mesurer l'exactitude et la fidélité, suivi sur MLflow, puis conteneurisation et déploiement.

## 📦 Livrables attendus

*   [ ] Application déployée et accessible en ligne.
*   [ ] Tableau des résultats d'évaluation détaillant les performances des agents.
*   [ ] Documentation avec schéma d'architecture et démo.
