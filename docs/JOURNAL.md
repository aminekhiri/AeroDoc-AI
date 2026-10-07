# Journal de bord AeroDoc-AI

Trace de ce qui a été fait dans le projet, dans l'ordre. À compléter au fil de l'eau.

## État actuel (7 octobre 2026)

- **Étape de la roadmap** : fin de l'étape 2 (Ingestion & RAG). La recherche fonctionne, sans LLM pour l'instant.
- **Corpus** : 4 PDF, 2551 pages, 5867 passages indexés.
- **Mesure de référence** (pages attendues auditées, voir section 8), recall@5 sur 15 questions :
  - `multilingual-e5-small` : **12/15 = 80,0 %**, MRR 0,622
  - `bge-m3` : **13/15 = 86,7 %**, MRR 0,756
- **Modèle d'embedding retenu : `BAAI/bge-m3`** (décision du 7 octobre, section 9). Le `.env` est déjà dessus. La table `e5-small` est conservée pour comparer plus tard.
- **À faire en priorité** : étape 3 (SQL et agents). Les questions q11 et q15, ratées par les deux modèles, sont mises de côté pour un futur reranker ou une recherche hybride.

---

## 1. Corpus (3 octobre)

- Création de `scripts/download_corpus.py` : télécharge les PDF listés dans `CORPUS` vers `data/raw/` et écrit `data/manifest.csv` (statut, taille, SHA-256).
- Airbus et EASA se téléchargent automatiquement. Safran, Thales et Latécoère (optionnel) se téléchargent à la main depuis la `page_url`.
- Documents présents en local dans `data/raw/` :
  - `airbus_urd_2023_en.pdf`
  - `easa_cs25_amendment25_en.pdf`
  - `SAFRAN_DEU_2025_US_PDF MEL_0204.pdf`
  - `Thales-Universal-Registration-Document-2025-EN.pdf`
- Latécoère n'a pas été téléchargé.
- Le script reste utile : les PDF ne sont pas dans Git, donc il sert à reconstituer `data/raw/` sur une autre machine.

## 2. Un premier découpage abandonné (3 octobre)

- J'avais écrit `scripts/extract_chunks.py` (PyMuPDF, découpage en passages, sortie JSONL).
- Il a été **supprimé** : `ingest.py` fait sa propre extraction et son propre découpage, et ne lisait jamais `chunks.jsonl`. Deux découpages indépendants, donc un doublon inutile.
- Un seul endroit découpe maintenant : `ingest.py`.

## 3. Nettoyage de Git (3 octobre)

- Les 4 PDF et `chunks.jsonl` avaient été commités par erreur. Ils ont été retirés du suivi Git, avec des `.gitkeep` pour garder le dossier `data/raw/`.
- Ajouts au `.gitignore` : fichiers de verrouillage Word (`~$*`), exception `!.env.example`.
- Historique local réécrit pour retirer les PDF et la ligne `Co-Authored-By: Claude` des commits. Règle pour la suite : **aucune mention de Claude dans les commits**.
- **Pas encore poussé** : `main` local et `origin/main` ont divergé.

## 4. Pipeline d'ingestion et de recherche (3 et 4 octobre)

Code dans `src/` :

- `config.py` : réglages partagés (modèle d'embedding, pgvector).
- `ingest.py` : PDF, pages, passages (LlamaIndex `SentenceSplitter`), embeddings locaux, stockage dans pgvector. Option `--reset` pour tout réindexer.
- `search.py` : affiche les top-k passages d'une question, avec leur score, leur fichier et leur page.

Base de données : Postgres + pgvector, lancée avec `docker compose up -d` (conteneur `aerodoc-pgvector`).

**Changement de modèle d'embedding** : `BAAI/bge-m3` (1024 dimensions) remplacé par `intfloat/multilingual-e5-small` (384 dimensions), parce que l'ingestion était trop lente.

- `max_length` limité à 512 tokens (au lieu de 8192).
- Taille de batch : 32.
- Préfixes E5 `query: ` et `passage: `.
- Choix automatique du device : cuda, puis mps, puis cpu.
- Obligation de relancer `ingest.py --reset` après tout changement de modèle (la dimension des vecteurs change).

**Résultat de l'ingestion (sur CPU)** : 4 PDF, 2551 pages, 5867 passages, 13 min 17 s.

- Embeddings : environ 9 min 40 s.
- Le reste (lecture des PDF, découpage, écriture en base) : environ 3 min 40 s sur cette exécution.

## 5. PyTorch avec GPU (7 octobre)

- La version installée au départ était `torch 2.14.1+cpu`, donc le GPU de la RTX 4060 Ti n'était pas utilisé.
- Installation de `torch 2.14.1+cu130` dans le `.venv`. L'installation a pris environ 25 minutes : 2 Go à télécharger sur une connexion lente.
- Vérification : `torch.cuda.is_available()` renvoie `True`.
- Gain mesuré ensuite sur le GPU, avec `e5-small` : ingestion complète en **182 s**, contre 796 s sur le CPU.
- À savoir : `requirements.txt` ne contient pas `torch`. Sur un autre appareil, il faut installer la bonne version (CUDA, CPU ou Mac).

## 6. Organisation des dossiers (7 octobre)

```
src/       config.py, ingest.py, search.py, eval_retrieval.py
scripts/   download_corpus.py
data/      raw/ (PDF, non suivis), manifest.csv, retrieval_questions.csv, eval/
docs/      roadmap, ce journal
```

Les commandes se lancent depuis la racine du projet, car les chemins (`data/raw`) y sont relatifs.

## 7. Identifiants hors du code (7 octobre)

- `docker-compose.yml` lit `PG_USER`, `PG_PASSWORD` et `PG_DB` depuis le `.env`, sans valeur par défaut.
- `src/config.py` n'a plus de valeur par défaut pour ces trois variables : il s'arrête avec un message clair si elles manquent.
- `.env.example` ne contient plus que des exemples (`<your_user>`, `<your_password>`…). Le vrai `.env` n'est jamais commité.
- Le conteneur local tourne encore avec l'ancien mot de passe. Pour changer, il faut `docker compose down -v` (efface les données) puis tout réindexer.

## 8. Évaluation du retrieval (7 octobre)

### Principe

- 15 questions écrites à la main (`data/retrieval_questions.csv`), avec le fichier et les pages où se trouve la réponse.
- Script `src/eval_retrieval.py` : une question est réussie (HIT) si un des k passages remontés vient du bon fichier **et** d'une des bonnes pages. Les pages comparées sont les numéros imprimés (`page_label`), pas l'index physique du PDF.
- Mesures : recall@5 (part des questions réussies) et MRR (1 = le bon passage est toujours en première position).
- Le nom du fichier Safran est normalisé (espace ou underscore) pour comparer.

Commande : `python src/eval_retrieval.py --verbose --out data/eval/<nom>.csv`

### Résultats bruts, avec la référence d'origine (e5-small)

| | recall@5 |
|---|---|
| **Total** | **8/15 = 53,3 %** (MRR 0,294) |
| Anglais / Français | 5/8 / 3/7 |
| Airbus / Thales / Safran / EASA | 2/4 / 3/4 / 0/3 / 3/4 |

Questions manquées : q01, q03, q05, q09, q10, q11, q15. Fichier : `data/eval/ref0_originale_e5-small_k5.csv`.

Problème identifié : la mesure est stricte sur une seule page. Pour q05 et q09, le modèle remontait une page qui contient bien la réponse, mais ce n'était pas la page notée dans le CSV.

### Première correction, à la main (q05, q09, nom du fichier Safran)

- q05 : ajout de la p.48. q09 : ajout des p.36 et p.69. Nom du fichier Safran corrigé dans le CSV.
- Limite de cette méthode : j'avais corrigé **seulement les questions dont j'avais vu le résultat des modèles**. Elle peut donc favoriser un modèle. D'où l'audit ci-dessous.

| Référence 1 | e5-small | bge-m3 |
|---|---|---|
| recall@5 | 10/15 = 66,7 % | 9/15 = 60,0 % |
| MRR | 0,428 | 0,439 |

Fichiers : `ref1_q05-q09_e5-small_k5.csv` et `ref1_q05-q09_bge-m3_k5.csv`.

### Audit de la référence, indépendant des modèles

Méthode, appliquée aux 15 questions de la même façon :

1. Pour chaque question, une ou deux chaînes-clés de la réponse (« 735 », « 8,598 », « 147,893 », « 3,257 », « 22,136 », « 84,958 », « 3.90 », « 1,802 », « 31,189 », « 90 seconds », « 2438 m », « 48 inches », « applicable to turbine »…). Pour q08, un motif plus large (R&D autofinancée, montants en milliards).
2. Recherche de ces chaînes dans **toutes** les pages du PDF attendu (texte extrait avec `pypdf`, comme LlamaIndex, avec le numéro de page imprimé). Aucun modèle d'embedding n'intervient.
3. Pour chaque page trouvée, lecture du contexte : la page est **valide** si elle donne ce chiffre pour le bon indicateur, la bonne année et la bonne entité. Elle est **rejetée** si le chiffre sert de dénominateur (ratios, taxonomie verte, intensité carbone), appartient à un autre indicateur ou est un faux positif (ex. « 735 » dans « 59,735 »).
4. Les pages valides sont ajoutées à `expected_page_labels`. Aucune page existante n'a été retirée : les 21 pages d'origine contiennent bien la réponse.
5. Le détail (question, page, décision, raison, extrait) est dans `data/eval/reference_audit.csv`.

Bilan : **96 pages examinées : 21 déjà attendues (confirmées), 32 ajoutées, 43 rejetées.**

| Question | Pages ajoutées |
|---|---|
| q01 Airbus livraisons | 44, 91, 178, 179, 250 |
| q03 Airbus effectif | 120, 138, 171 |
| q04 Airbus R&D | 91, 180 |
| q05 Thales ventes | 42, 43, 49, 51, 256, 262, 264 |
| q06 Thales effectif | 152, 192, 283 |
| q07 Thales dividende | 52, 250, 282, 301, 327 |
| q08 Thales R&D | 39, 54 |
| q10 Safran Propulsion | 29 |
| q11 Safran chiffre d'affaires | 66, 199, 213, 216 |

Inchangées : q02, q09, q12, q13, q14, q15.

Choix qui relèvent d'un jugement (à relire si besoin) : q01 p.250 acceptée (le fait est énoncé dans un passage sur la rémunération) ; q11 p.66 et p.213 acceptées (tableaux de réconciliation, chiffre d'affaires consolidé) ; q06 p.193 rejetée (libellé de ligne « part-time » ambigu) ; q05 p.54, 173, 184 et 263 rejetées (ventes utilisées comme dénominateur).

### Résultats avec la référence auditée (référence 2)

| | e5-small | bge-m3 |
|---|---|---|
| **recall@5** | **12/15 = 80,0 %** | **13/15 = 86,7 %** |
| **MRR** | 0,622 | 0,756 |
| Bon passage en 1re position | 7/15 | 10/15 |
| Anglais | 7/8 | 7/8 |
| Français | 5/7 | 6/7 |
| Airbus | 3/4 | 4/4 |
| Thales | 4/4 | 4/4 |
| Safran | 2/3 | 2/3 |
| EASA CS-25 | 3/4 | 3/4 |
| Questions manquées | q03, q11, q15 | q11, q15 |

Fichiers : `ref2_auditee_e5-small_k5.csv` et `ref2_auditee_bge-m3_k5.csv`.

Récapitulatif des trois versions de la référence :

| Référence | e5-small | bge-m3 |
|---|---|---|
| 0, d'origine | 8/15 (53,3 %), MRR 0,294 | non mesuré |
| 1, + q05 et q09 | 10/15 (66,7 %), MRR 0,428 | 9/15 (60,0 %), MRR 0,439 |
| 2, auditée | 12/15 (80,0 %), MRR 0,622 | 13/15 (86,7 %), MRR 0,756 |

Lecture :
- L'écart de recall entre les deux modèles est **d'une seule question** (6,7 points) et change de sens entre les références 1 et 2 : il n'est pas significatif.
- L'écart de rang est plus net avec la référence 2 : `bge-m3` met le bon passage en 1re position 10 fois sur 15, contre 7 pour `e5-small`.
- q11 (chiffre d'affaires consolidé de Safran, 5 pages valides) et q15 (domaine d'application de CS-25, 1 page) sont ratées par les deux modèles.
- Avec 15 questions, une question vaut 6,7 points. Il faudra plus de questions (la roadmap prévoit 30 à 40) avant de trancher sur de petits écarts.

### Détails de l'expérience

- **Deux tables dans pgvector**, pour comparer sans réindexer : `data_chunks` (bge-m3, 1024 dimensions, table par défaut) et `data_chunks_e5` (e5-small, 384 dimensions). Les variables d'environnement surchargent le `.env` :
  ```
  PG_TABLE=chunks_e5 EMBED_MODEL=intfloat/multilingual-e5-small EMBED_DIM=384 python src/eval_retrieval.py
  ```
- **Durées sur le GPU** : e5-small, 182 s au total ; bge-m3, environ 4 min 20 s d'embeddings (820 s au total, téléchargement du modèle de 2,3 Go compris).
- **Incident** : le premier essai avec bge-m3 a échoué, disque plein (2 Go libres) pendant le téléchargement du modèle. Le dossier de cache du modèle existait mais était vide, ce qui avait fait croire à tort qu'il était déjà téléchargé. La base n'avait pas été touchée. Nouvel essai réussi après avoir libéré de la place (14 Go libres). Sous Windows sans mode développeur, le cache Hugging Face n'utilise pas de liens symboliques et prend plus de place.

## 9. Décision : `bge-m3` retenu (7 octobre)

**Choix : `BAAI/bge-m3`** (1024 dimensions), table pgvector `data_chunks`. Le `.env` est déjà configuré dessus.

Raisons :
- **Meilleur classement** : le bon passage est en 1re position 10 fois sur 15 (contre 7), MRR 0,756 (contre 0,622).
- **Meilleur en français** : 6/7 (contre 5/7) ; en anglais, les deux font 7/8. Les documents sont surtout en anglais, mais les questions sont souvent posées en français.
- **Coût de production jugé acceptable.**

Réserves, pour garder la mémoire de ce qui n'est pas prouvé :
- L'écart de **recall@5** n'est que d'une question (13/15 contre 12/15) et change de sens selon la version de la référence : la décision repose sur le **classement** (MRR, 1re position), pas sur le recall.
- Le coût en production n'a pas été mesuré : le modèle pèse 2,3 Go (contre 470 Mo) et la latence d'une question sur CPU, pour Cloud Run, n'a pas été testée.

Autres décisions :
- **La table `data_chunks_e5` (e5-small) est conservée.** Elle ne coûte rien et permettra de comparer les deux modèles avec un reranker.
- **q11 et q15 sont laissées de côté** : candidates naturelles pour un reranker ou une recherche hybride, à tester plus tard. Elles ne bloquent pas la suite.

---

## Pistes pour améliorer le retrieval

- Tester un **reranker** et/ou une **recherche hybride** (vectorielle et mots-clés), en commençant par q11 et q15, avec `bge-m3` puis `e5-small` pour comparer.
- Filtrer les passages « bruit » (en-têtes de page, numéros) : ils remontent dans les résultats et le filtre actuel (plus de 50 caractères) les laisse passer.
- Agrandir le jeu de questions (30 à 40 dans la roadmap) pour pouvoir départager les modèles.
- Essayer d'autres tailles de passages (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
- Remplacer le lecteur de PDF par PyMuPDF (plus rapide, et probablement meilleur sur les tableaux).
- Ajouter une recherche par mots-clés en plus de la recherche vectorielle, utile pour les chiffres précis.

## Reste à faire

- [ ] Pousser `main` : `git push --force-with-lease=main:188d06a origin main`.
- [ ] Supprimer la branche `feature/extract-file` sur GitHub (elle contient encore les PDF) et les branches locales `feature/extract-file` et `backup/avant-nettoyage`.
- [x] Commiter `src/eval_retrieval.py`, `data/retrieval_questions.csv` et `data/eval/`.
- [x] Choisir le modèle d'embedding : `bge-m3` retenu (section 9). `.env` et base alignés.
- [ ] Aligner les valeurs par défaut sur `bge-m3` : `src/config.py` et `.env.example` indiquent encore `e5-small` (384 dimensions). Quelqu'un qui clone le dépôt aurait un autre modèle que le vôtre.
- [ ] Reranker ou recherche hybride, pour q11 et q15 (plus tard).
- [ ] Étape 3 : Text-to-SQL avec DuckDB, agents LangGraph (routeur, rédacteur, vérificateur).
- [ ] Étape 4 : évaluation complète avec MLflow, API FastAPI, Docker, déploiement sur Cloud Run (base Postgres hébergée, PyTorch CPU dans l'image, modèle inclus dans l'image).

## Historique des commits (local)

```
(ce commit) docs: record bge-m3 as the chosen embedding model
e7f2bc3 eval: baseline results e5-small vs bge-m3
9dc7c21 eval: audit reference pages (model-independent search)
d6790a5 eval: add retrieval evaluation script
48b2a37 chore: remove hardcoded database credentials
77bb476 refactor: organize project into src/, scripts/ and docs/
2561882 perf: switch to multilingual-e5-small and cap embedding length
24104e7 add: LlamaIndex ingestion and search pipeline (bge-m3 + pgvector)
2ded446 chore: remove unused extract_chunks step
9e38b27 chore: ignore Office lock files
b7cd07b chore: stop tracking data/raw and data/processed
dbbbc5e add: exract chunk of the pdf files
d0c2814 add: extraction of the files for the KB
4e74d10 add: initialzation of the README file
```
