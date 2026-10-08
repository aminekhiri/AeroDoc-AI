# Journal de bord AeroDoc-AI

Trace de ce qui a été fait dans le projet, dans l'ordre. À compléter au fil de l'eau.

## État actuel (8 octobre 2026)

- **Étape de la roadmap** : fin de l'étape 2 (Ingestion & RAG) : la recherche fonctionne et le LLM (Gemini) écrit des réponses avec citations (`src/ask.py`, section 10). Pas encore de routeur ni de SQL.
- **Corpus** : 4 PDF, 2551 pages, 5867 passages indexés.
- **Mesure de référence** (pages attendues auditées, voir section 8), recall@5 sur 15 questions :
  - `multilingual-e5-small` : **12/15 = 80,0 %**, MRR 0,622
  - `bge-m3` : **13/15 = 86,7 %**, MRR 0,756
- **Modèle d'embedding retenu : `BAAI/bge-m3`** (décision du 7 octobre, section 9). Le `.env` est déjà dessus. La table `e5-small` est conservée pour comparer plus tard.
- **LLM** : Google Gemini, modèle **`gemini-3.5-flash-lite`** retenu pour l'instant (limites larges, bien pour les tests ; Mistral abandonné). Voir sections 10 et 11.
- **Évaluation des réponses** (15 questions + 2 hors corpus, section 11) : **13/15 OK** ; les 2 autres sont les misses de la recherche (q11 : refus honnête, q15 : réponse partielle). Refus hors corpus : 2/2 corrects (en deux lancements différents).
- **À faire en priorité** : corriger le prompt pour que les refus ne citent aucune source, puis étape 3 (SQL et agents). Les questions q11 et q15, ratées par les deux modèles, sont mises de côté pour un futur reranker ou une recherche hybride.

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
src/       config.py, ingest.py, search.py, ask.py, eval_retrieval.py, eval_answers.py
scripts/   download_corpus.py
data/      raw/ (PDF, non suivis), manifest.csv, retrieval_questions.csv, out_of_scope_questions.csv, eval/
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

## 10. Génération de réponses avec un LLM (7 et 8 octobre)

### Ce qui a été fait

Nouveau script `src/ask.py` : une question, puis

1. recherche des 5 passages les plus proches dans pgvector (avec `bge-m3`) ;
2. construction d'un prompt avec ces passages numérotés `[1]`, `[2]`… (document et page) ;
3. appel au LLM, qui écrit la réponse ;
4. affichage de la réponse, puis des sources avec leur score. Une `*` marque celles que la réponse a citées.

Règles du prompt : répondre **uniquement** à partir des sources, citer `[n]` après chaque information, reprendre les chiffres avec leur unité et leur année, répondre dans la langue de la question, et dire « Je ne trouve pas cette information dans les documents » si les sources ne contiennent pas la réponse (c'est la promesse du README).

Options : `--k` (nombre de passages donnés au LLM) et `--show-prompt` (affiche le prompt sans appeler le LLM, utile pour déboguer sans clé).

```
.venv\Scripts\Activate.ps1
python src/ask.py "Combien d'avions Airbus a-t-il livrés en 2023 ?"
```

Il faut activer le `.venv` (sinon : `No module named 'llama_index'`) et lancer Docker (`docker compose up -d`).

### Mistral abandonné, Gemini retenu

- Première version avec **Mistral** (commit `6ba7861`, dont le message parle de « api key » : aucune clé n'y est, elle reste dans le `.env`). Premier appel refusé : **erreur 429 « Rate limit exceeded »**, une limite du compte (forfait gratuit probable), pas un bug du code.
- Passage à **Google Gemini** (paquet `llama-index-llms-google-genai`). Le code était déjà centralisé : seuls les noms ont changé. `MISTRAL_API_KEY` devient `GEMINI_API_KEY` ; `LLM_MODEL`, `LLM_TEMPERATURE` et `LLM_MAX_TOKENS` ne changent pas de nom. Paquets Mistral désinstallés.
- Modèle : `gemini-3.7-flash` d'abord (défaut du paquet), qui a répondu **503 « high demand »** après ses 3 essais automatiques (surcharge côté Google, passagère). Remplacé par **`gemini-3.8-flash`**, qui répond. Le modèle se change dans le `.env`, sans toucher au code. *(Le modèle retenu a ensuite changé : `gemini-3.5-flash-lite`, section 11.)*

### Points techniques à retenir

- Créer le client Gemini **fait déjà un appel réseau** : la bibliothèque vérifie que le modèle existe. Une clé ou un nom de modèle faux échoue donc dès le début, et le message d'erreur est expliqué par `ask.py`.
- Les modèles `gemini-3` peuvent refuser une température imposée : le code ne l'envoie pas pour eux.
- `LLM_MAX_TOKENS` passe de 700 à 2048 : les tokens de « réflexion » de Gemini comptent dans la limite et pourraient couper la réponse.
- La bibliothèque réessaie 3 fois toute seule en cas d'erreur : le mécanisme de réessai écrit à la main a été supprimé.
- Conseils affichés selon l'erreur : quota (429), modèle introuvable (404), clé refusée (400, 401, 403).
- Le premier lancement semble figé : le chargement de `transformers` prend 10 à 60 s sans rien afficher. Des messages de progression ont été ajoutés.

### Tests faits (2 questions, à la main)

| Question | Résultat |
|---|---|
| « Combien d'avions Airbus a-t-il livrés en 2023 ? » | **735 avions commerciaux**, répartis en A220 : 68, A320 : 571, A330 : 32, A350 : 64, plus 8 A400M. Les chiffres ont été relus dans le PDF (p.44 et p.171) : exacts, et 68 + 571 + 32 + 64 = 735. Les citations pointent vers les bonnes pages. |
| « Quel est le prix de l'action Boeing aujourd'hui ? » (hors corpus) | « Je ne trouve pas cette information dans les documents. » Aucune invention, aucune source citée. Les scores des passages remontés sont bas (0,52 à 0,56, contre 0,69 à 0,73 pour une question avec réponse). |

### Limites

- Seulement **2 questions** testées à la main à ce stade. *(Mesure systématique faite ensuite : section 11.)*
- Chaque question envoie la question et 5 passages à l'API de Google. Sans risque pour des documents publics, à reconsidérer pour des documents confidentiels.
- Idée non faite : un **seuil de score** en dessous duquel on répond « je ne sais pas » sans appeler le LLM (économise un appel et supprime le risque d'invention). À calibrer sur davantage de questions.

## 11. Évaluation des réponses du LLM (8 octobre)

### Le script `src/eval_answers.py`

Il mesure la **réponse** du LLM (`eval_retrieval.py` ne mesurait que la recherche). Une question = **un appel LLM**, avec la même recherche et le même prompt qu'`ask.py`. Les contrôles sont automatiques, sans deuxième LLM comme juge :

- **Question du corpus** : la réponse contient l'un des mots-clés de la colonne `answer_keys` de `data/retrieval_questions.csv`, **et** cite (`[n]`) une source qui vient d'une page attendue (la référence auditée de la section 8).
- **Question hors corpus** (`data/out_of_scope_questions.csv`) : la réponse dit « je ne trouve pas ».
- Les mots-clés tolèrent le formatage des nombres : `22_136|22_1` accepte « 22,136 », « 22 136 » et « 22,1 » (`_` = un séparateur, `~` = séparateur facultatif, `|` = alternatives).
- Garde-fous : plafond dur d'appels (`--max-calls`, 10 par défaut, vérifié **avant** tout appel), pause entre les appels, arrêt après 2 erreurs d'API de suite, et `--dry-run` pour tester la recherche sans rien dépenser.

```
python src/eval_answers.py --dry-run                       # sans appel LLM
python src/eval_answers.py                                 # 8 questions + 2 hors corpus = 10 appels
python src/eval_answers.py --ids q01,q02,...,q15 --max-calls 17 --out data/eval/<nom>.csv
```

Verdicts : `OK`, `BONNE REPONSE, CITATION FAUSSE`, `REPONSE FAUSSE`, `REFUS (page non retrouvee)` (le modèle refuse parce que la recherche n'a pas ramené la bonne page : ce n'est pas une faute du LLM), `REFUS A TORT`, `N'A PAS REFUSE`, `ERREUR API`.

### Historique des essais (3 modèles Gemini)

| Essai | Questions évaluées | Résultat | Fichier dans `data/eval/` |
|---|---|---|---|
| `gemini-3.8-flash`, 10 questions | 3/10 | 3/3 OK ; **503 « high demand »** sur q01, q09, q11, arrêt du script | `answers_gemini-3.8-flash_2026-10-08.csv` |
| `gemini-3.6-flash`, 10 questions | 6/10 | 5 OK + q11 refusée ; **429 « quota dépassé »** sur q12, q13, arrêt | `answers_gemini-3.6-flash_2026-10-08.csv` |
| `gemini-3.5-flash-lite`, 10 questions | **10/10** | **7/8** du corpus OK + **2/2** refus corrects | `answers_gemini-3.5-flash-lite_2026-10-08.csv` |
| `gemini-3.5-flash-lite`, 15 + 2 questions | 16/17 | voir ci-dessous ; **429** sur le 17e appel (o02) | `answers_gemini-3.5-flash-lite_15q_2026-10-08.csv` |

Les essais `3.8` et `3.6` se sont arrêtés sur des problèmes de capacité ou de quota, pas sur de mauvaises réponses : on ne peut donc pas comparer leur qualité à celle de `3.5-flash-lite`.

### Résultats finaux (`gemini-3.5-flash-lite`, 15 questions du corpus + 2 hors corpus)

- **13/15 OK** (bonne réponse et bonne citation) : q01 à q10, q12, q13, q14.
- **q11 : refus** (« Je ne trouve pas cette information »). La recherche n'avait pas remonté la bonne page (miss connu des deux modèles d'embedding). Le comportement du LLM est correct.
- **q15 : réponse partielle, citation fausse.** Le PDF dit « applicable to **turbine powered** Large Aeroplanes » ; le modèle répond « aux grands aéronefs » sans le « turbine powered », et ne cite pas la p.51 (autre miss de la recherche).
- **Hors corpus : o01 (Boeing) refusée correctement.** o02 (Ligue des champions) : 429 à ce lancement, mais refusée correctement lors du lancement à 10 questions avec le même modèle : **2 refus corrects sur 2** au total, jamais dans le même lancement.
- Vérifié à la main contre les PDF : q08 (1 328 M€ d'autofinancement R&D, cohérent avec « over €1 billion »), q10 (65 % de services), q12, q13, q14.

Les deux questions non réussies sont **les deux misses de la recherche** (q11, q15) : le LLM ne s'est pas trompé sur ce qu'il avait, c'est la recherche qui ne lui a pas donné la bonne page.

### Défauts constatés

- **Les refus citent des sources** : « Je ne trouve pas cette information dans les documents [1][2][3][4][5] » (q11, o01). Citer cinq sources en refusant est trompeur. À corriger dans le prompt (« en cas de refus, ne cite aucune source »). Le contrôle automatique ne le pénalise pas.
- **Le mot-clé de q15 est trop indulgent** : `large aeroplane|grands avions` accepte une réponse qui oublie « turbine powered ». Le verdict « BONNE REPONSE » de q15 est donc généreux ; la réponse est en réalité partielle.
- **Le plafond compte les questions, pas les requêtes HTTP.** Le client Gemini renvoie lui-même une requête quand Google répond 503 (jusqu'à 3 essais). Lors de l'essai avec `3.8`, 6 questions ont généré une quinzaine de requêtes. Pour un plafond exact, il faudrait désactiver ces relances pendant l'évaluation (`max_retries=0`).
- **Quotas** : un 429 est tombé sur le dernier appel d'un lancement de 17 appels. Hypothèse non vérifiée : limite par minute. À contrôler sur https://ai.dev/rate-limit ; on peut augmenter `--pause`.

### Décision

**`gemini-3.5-flash-lite` est retenu pour l'instant** : ses limites sont larges, ce qui convient aux tests et aux évaluations, et c'est le seul qui a permis des mesures complètes. Valeur par défaut dans `src/config.py` et `.env.example`. La question sera à rouvrir pour la production, avec un modèle plus capable si la qualité des réponses l'exige.

---

## Pistes pour améliorer le retrieval

- Corriger le **prompt** : en cas de refus, ne citer aucune source.
- Ajouter un **seuil de score** : en dessous, répondre « je ne sais pas » sans appeler le LLM (à calibrer sur plus de questions).
- Tester un **reranker** et/ou une **recherche hybride** (vectorielle et mots-clés), en commençant par q11 et q15, avec `bge-m3` puis `e5-small` pour comparer.
- Filtrer les passages « bruit » (en-têtes de page, numéros) : ils remontent dans les résultats et le filtre actuel (plus de 50 caractères) les laisse passer.
- Agrandir le jeu de questions (30 à 40 dans la roadmap) pour pouvoir départager les modèles.
- Essayer d'autres tailles de passages (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
- Remplacer le lecteur de PDF par PyMuPDF (plus rapide, et probablement meilleur sur les tableaux).
- Ajouter une recherche par mots-clés en plus de la recherche vectorielle, utile pour les chiffres précis.

## Reste à faire

- [x] Branche `feature/extract-file` supprimée (GitHub et local) : les PDF ne sont plus sur le dépôt distant.
- [ ] Pousser les derniers commits (`main` est en avance sur GitHub) : `git push origin main`.
- [x] Commiter `src/eval_retrieval.py`, `data/retrieval_questions.csv` et `data/eval/`.
- [x] Choisir le modèle d'embedding : `bge-m3` retenu (section 9). `.env` et base alignés.
- [x] Valeurs par défaut alignées sur `bge-m3` (1024 dimensions) dans `src/config.py` et `.env.example`. Le modèle `e5-small` reste indiqué en commentaire comme alternative légère.
- [x] Brancher un LLM (Gemini) et générer des réponses avec citations (section 10).
- [x] Évaluer les réponses du LLM : 15 questions + 2 hors corpus (section 11).
- [ ] Prompt : un refus ne doit citer aucune source.
- [ ] Durcir le mot-clé de q15 (« turbine powered ») et relancer o02 dans le même lancement que les autres.
- [ ] Reranker ou recherche hybride, pour q11 et q15 (plus tard).
- [ ] Étape 3 : Text-to-SQL avec DuckDB, agents LangGraph (routeur, rédacteur, vérificateur).
- [ ] Étape 4 : évaluation complète avec MLflow, API FastAPI, Docker, déploiement sur Cloud Run (base Postgres hébergée, PyTorch CPU dans l'image, modèle inclus dans l'image).

## Historique des commits (local)

```
(ce commit) eval: answer evaluation results (gemini-3.5-flash-lite)
c8cf6ac config: default LLM to gemini-3.5-flash-lite
c4f6dfd eval: add LLM answer evaluation script
9123250 docs: journal, LLM generation with Gemini
bb1cd00 feat: switch answer generation from Mistral to Gemini
6ba7861 add: implmentation of mistral LLM + api key
18fe1a1 config: default to bge-m3
d0c7e68 docs: record bge-m3 as the chosen embedding model
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
