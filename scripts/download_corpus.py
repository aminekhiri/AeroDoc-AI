#!/usr/bin/env python3
"""
Télécharge le corpus de documents publics et génère data/manifest.csv.

Usage :
    pip install requests
    python scripts/download_corpus.py

- Les documents avec une URL directe sont téléchargés dans data/raw/.
- Les autres (url = None) sont à télécharger à la main depuis `page_url`,
  puis à placer dans data/raw/ sous le nom indiqué dans `filename`.
- Relancer le script met à jour le manifeste (aucun fichier valide n'est re-téléchargé).

Les chemins sont relatifs au répertoire courant : lancer le script depuis la
racine du projet.
"""

import csv
import hashlib
import sys
from pathlib import Path

import requests

# Dossier de destination des PDF bruts.
RAW_DIR = Path("data/raw")
# Fichier CSV qui recense l'état de chaque document du corpus.
MANIFEST = Path("data/manifest.csv")
# Certains sites refusent les requêtes sans User-Agent de navigateur.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; corpus-downloader)"}
# Délai maximal (en secondes) pour la connexion et la lecture HTTP.
TIMEOUT = 60

# Description du corpus. Chaque entrée contient :
#   id        : identifiant unique du document
#   company   : émetteur (entreprise ou autorité)
#   doc_type  : "annual_report" ou "regulation"
#   year      : année de l'exercice / de la publication
#   language  : code langue ISO 639-1
#   filename  : nom du fichier dans data/raw/
#   url       : lien direct vers le PDF, ou None si téléchargement manuel
#   page_url  : page web où trouver le document
#   optional  : True si le document n'est pas indispensable au corpus
CORPUS = [
    {
        "id": "airbus_urd_2023",
        "company": "Airbus",
        "doc_type": "annual_report",
        "year": 2023,
        "language": "en",
        "filename": "airbus_urd_2023_en.pdf",
        "url": "https://www.airbus.com/sites/g/files/jlcbta136/files/2024-03/Airbus-Universal-Registration-Document-2023.pdf",
        "page_url": "https://www.airbus.com/en/investors",
        "optional": False,
    },
    {
        "id": "easa_cs25_amdt25",
        "company": "EASA",
        "doc_type": "regulation",
        "year": 2021,
        "language": "en",
        "filename": "easa_cs25_amendment25_en.pdf",
        "url": "https://www.easa.europa.eu/en/downloads/129017/en",
        "page_url": "https://www.easa.europa.eu/en/document-library/easy-access-rules",
        "optional": False,
    },
    {
        "id": "safran_urd_2025",
        "company": "Safran",
        "doc_type": "annual_report",
        "year": 2025,
        "language": "en",
        "filename": "safran_urd_2025_en.pdf",
        "url": None,
        "page_url": "https://www.safran-group.com/fr/finance/publications-resultats",
        "optional": False,
    },
    {
        "id": "thales_urd_2025",
        "company": "Thales",
        "doc_type": "annual_report",
        "year": 2025,
        "language": "en",
        "filename": "thales_urd_2025_en.pdf",
        "url": None,
        "page_url": "https://www.thalesgroup.com",
        "optional": False,
    },
    {
        "id": "latecoere_urd_2025",
        "company": "Latecoere",
        "doc_type": "annual_report",
        "year": 2025,
        "language": "fr",
        "filename": "latecoere_urd_2025_fr.pdf",
        "url": None,
        "page_url": "https://www.latecoere.aero/finance/rapports-annuels-et-semestriels/",
        "optional": True,
    },
]

# Colonnes du manifeste, dans l'ordre d'écriture.
FIELDS = [
    "id", "company", "doc_type", "year", "language", "filename",
    "source_url", "page_url", "optional", "status", "size_mb", "sha256",
]


def sha256_of(path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier.

    Le fichier est lu par blocs de 1 Mio pour ne pas charger un PDF de
    plusieurs centaines de Mo en mémoire. L'empreinte permet de vérifier
    qu'un document n'a pas changé entre deux exécutions.

    Args:
        path: chemin du fichier à hacher (doit exister et être lisible).

    Returns:
        L'empreinte sous forme de chaîne hexadécimale de 64 caractères.

    Raises:
        OSError: si le fichier n'existe pas ou ne peut pas être lu.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_pdf(path: Path) -> bool:
    """Indique si un fichier est un PDF en lisant sa signature.

    Un PDF valide commence toujours par les 5 octets ``%PDF-``. Ce test
    détecte les cas où un serveur renvoie une page HTML (erreur, captcha,
    redirection) à la place du document attendu.

    Args:
        path: chemin du fichier à tester.

    Returns:
        True si le fichier commence par ``%PDF-``, False sinon, ou si le
        fichier est absent ou illisible (aucune exception n'est levée).
    """
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def download(url: str, dest: Path) -> None:
    """Télécharge un PDF depuis une URL vers un fichier local.

    Le contenu est écrit en streaming dans un fichier temporaire
    ``<dest>.part``. Ce n'est qu'une fois le téléchargement terminé et le
    format vérifié que le fichier est renommé en ``dest``. Ainsi, une
    interruption ou un contenu invalide ne laisse jamais un faux PDF à la
    place du document final.

    Args:
        url: adresse directe du PDF.
        dest: chemin final du fichier (son dossier parent doit exister).

    Returns:
        None. En cas de succès, ``dest`` contient le PDF téléchargé
        (un fichier existant est écrasé).

    Raises:
        requests.HTTPError: si le serveur répond avec un code 4xx/5xx.
        requests.RequestException: en cas d'erreur réseau ou de timeout.
        ValueError: si le contenu reçu n'est pas un PDF (le fichier
            temporaire est alors supprimé).
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    if not is_pdf(tmp):
        tmp.unlink(missing_ok=True)
        raise ValueError("le fichier téléchargé n'est pas un PDF")
    tmp.replace(dest)


def process(doc: dict) -> dict:
    """Prépare un document du corpus et renvoie sa ligne de manifeste.

    Logique appliquée, dans l'ordre :
      1. le PDF est déjà présent et valide dans data/raw/  -> statut "ok"
         (pas de nouveau téléchargement) ;
      2. sinon, si une URL directe existe, on le télécharge -> "ok" en cas
         de succès, "error: <message>" en cas d'échec ;
      3. sinon (url = None) -> statut "manual", et les instructions de
         téléchargement manuel sont affichées.

    Les erreurs de téléchargement sont capturées : un document en échec
    n'interrompt pas le traitement des autres.

    Args:
        doc: une entrée de ``CORPUS`` (voir la description des clés plus haut).

    Returns:
        Un dictionnaire dont les clés sont exactement ``FIELDS``.
        ``size_mb`` (taille en Mo, arrondie à 2 décimales) et ``sha256``
        ne sont renseignés que si le statut est "ok" ; ils valent "" sinon.
    """
    dest = RAW_DIR / doc["filename"]
    status = ""

    if dest.exists() and is_pdf(dest):
        status = "ok"
    elif doc["url"]:
        print(f"[download] {doc['id']} ...")
        try:
            download(doc["url"], dest)
            status = "ok"
        except Exception as exc:  # réseau, HTTP, format
            status = f"error: {exc}"
            print(f"  -> échec : {exc}")
    else:
        status = "manual"

    if status == "manual":
        print(
            f"[manuel]   {doc['id']} : télécharger depuis\n"
            f"           {doc['page_url']}\n"
            f"           puis enregistrer sous {dest}"
        )

    row = {
        "id": doc["id"],
        "company": doc["company"],
        "doc_type": doc["doc_type"],
        "year": doc["year"],
        "language": doc["language"],
        "filename": doc["filename"],
        "source_url": doc["url"] or "",
        "page_url": doc["page_url"],
        "optional": doc["optional"],
        "status": status,
        "size_mb": "",
        "sha256": "",
    }
    if status == "ok":
        row["size_mb"] = round(dest.stat().st_size / 1e6, 2)
        row["sha256"] = sha256_of(dest)
    return row


def main() -> int:
    """Point d'entrée : traite tout le corpus et écrit le manifeste.

    Étapes :
      1. crée data/raw/ et data/ si nécessaire ;
      2. appelle ``process`` pour chaque document de ``CORPUS`` ;
      3. réécrit entièrement data/manifest.csv (UTF-8, une ligne par document) ;
      4. affiche un résumé et la liste des documents obligatoires
         (``optional = False``) qui ne sont pas encore prêts.

    Returns:
        Code de sortie du programme : toujours 0. Les documents manquants
        sont signalés à l'écran mais ne font pas échouer le script.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    rows = [process(doc) for doc in CORPUS]

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\nManifeste écrit : {MANIFEST} ({ok}/{len(rows)} documents prêts)")
    missing = [r["id"] for r in rows if r["status"] != "ok" and not r["optional"]]
    if missing:
        print("Documents obligatoires manquants :", ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
