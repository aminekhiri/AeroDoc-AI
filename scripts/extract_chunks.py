#!/usr/bin/env python3
"""
Phase 1 : extraction du texte des PDF et découpage en passages (chunks).

Usage :
    pip install pymupdf
    python scripts/extract_chunks.py
    python scripts/extract_chunks.py --chunk-size 1200 --overlap 150

Entrée  : data/manifest.csv + les PDF dans data/raw/
Sortie  : data/processed/chunks.jsonl (un passage par ligne, avec document et page)

Chaque passage reste dans une seule page, pour pouvoir citer "document, page N".
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import pymupdf as fitz  # PyMuPDF (versions récentes)
except ImportError:
    try:
        import fitz  # anciennes versions
    except ImportError:
        sys.exit("PyMuPDF est manquant. Installe-le avec : pip install pymupdf")

# Dossier des PDF bruts (rempli par download_corpus.py).
RAW_DIR = Path("data/raw")
# Manifeste produit par download_corpus.py : métadonnées de chaque document.
MANIFEST = Path("data/manifest.csv")
# Dossier et fichier de sortie des passages.
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "chunks.jsonl"

MIN_PAGE_CHARS = 80    # en dessous, la page est considérée vide (image, page blanche)
MIN_CHUNK_CHARS = 60   # en dessous, le passage est ignoré (numéros de page, titres isolés)


def sha256_of(path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier.

    Le fichier est lu par blocs de 1 Mio pour limiter la mémoire utilisée.
    L'empreinte est recopiée dans chaque passage (``doc_sha256``) afin de
    savoir de quelle version exacte du PDF il provient.

    Args:
        path: chemin du fichier à hacher (doit exister et être lisible).

    Returns:
        L'empreinte sous forme de chaîne hexadécimale de 64 caractères.

    Raises:
        OSError: si le fichier n'existe pas ou ne peut pas être lu.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(text: str) -> str:
    """Nettoie le texte brut extrait d'une page PDF.

    Transformations appliquées, dans l'ordre :
      1. suppression des traits d'union conditionnels (U+00AD) ;
      2. recollage des mots coupés en fin de ligne (``aéro-\\nnautique`` ->
         ``aéronautique``), seulement si la ligne suivante commence par une
         minuscule, pour ne pas fusionner des tirets légitimes ;
      3. ``\\r`` converti en ``\\n`` ;
      4. plusieurs sauts de ligne consécutifs réduits à un seul séparateur de
         paragraphe ``\\n\\n`` ;
      5. sauts de ligne simples (retours à la ligne visuels du PDF)
         remplacés par un espace ;
      6. espaces, tabulations et espaces insécables consécutifs réduits à
         un seul espace.

    Args:
        text: texte brut renvoyé par PyMuPDF pour une page.

    Returns:
        Le texte nettoyé, sans espaces en début ni en fin. Les paragraphes
        restent séparés par ``\\n\\n`` (utilisé par ``split_sentences``).
    """
    text = text.replace("­", "")          # trait d'union conditionnel
    text = re.sub(r"-\n(?=[a-zà-ÿ])", "", text)  # mots coupés en fin de ligne
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # retours à la ligne simples -> espace
    text = re.sub(r"[ \t ]+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Découpe un texte nettoyé en phrases.

    Le texte est d'abord séparé en paragraphes (``\\n\\n``), puis chaque
    paragraphe est coupé après un signe de ponctuation ``. ! ? ; :`` suivi
    d'espaces. La ponctuation reste attachée à la phrase qui la précède.
    Ce découpage est volontairement simple : des abréviations comme
    ``M. Dupont`` ou ``fig. 3`` peuvent provoquer une coupure.

    Args:
        text: texte issu de ``clean_text``.

    Returns:
        La liste des phrases dans l'ordre du texte, sans espaces superflus
        et sans élément vide.
    """
    parts = []
    for paragraph in text.split("\n\n"):
        parts.extend(re.split(r"(?<=[.!?;:])\s+", paragraph))
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Regroupe les phrases d'un texte en passages de taille bornée.

    Algorithme :
      - les phrases sont ajoutées une à une au passage courant tant que sa
        longueur reste inférieure ou égale à ``size`` ;
      - quand la phrase suivante ferait dépasser ``size``, le passage courant
        est enregistré et le suivant commence par ses ``overlap`` derniers
        caractères (coupés au premier espace pour ne pas commencer au milieu
        d'un mot), puis la nouvelle phrase. Ce chevauchement conserve du
        contexte entre deux passages consécutifs ;
      - une phrase plus longue que ``size`` est découpée de force en
        tranches de ``size`` caractères qui se chevauchent de ``overlap``.

    Un passage formé d'un chevauchement suivi d'une phrase peut dépasser
    légèrement ``size``.

    Args:
        text: texte nettoyé d'une seule page.
        size: taille maximale visée d'un passage, en caractères (> 0).
        overlap: nombre de caractères repris du passage précédent
            (0 <= overlap < size).

    Returns:
        La liste des passages, dans l'ordre. Les passages de moins de
        ``MIN_CHUNK_CHARS`` caractères sont supprimés.
    """
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        # phrase plus longue qu'un passage : découpe forcée
        while len(sentence) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:size])
            sentence = sentence[max(size - overlap, 1):]
        if current and len(current) + 1 + len(sentence) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            if " " in tail:
                tail = tail[tail.find(" ") + 1:]
            current = (tail + " " + sentence).strip()
        else:
            current = (current + " " + sentence).strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def read_manifest() -> list[dict]:
    """Lit le manifeste du corpus.

    Returns:
        Une liste de dictionnaires, un par ligne de data/manifest.csv, dont
        les clés sont les colonnes du CSV. Toutes les valeurs sont des
        chaînes (``year`` vaut par exemple ``"2023"``).

    Raises:
        SystemExit: si le manifeste n'existe pas. Le message invite à lancer
            download_corpus.py d'abord.
    """
    if not MANIFEST.exists():
        sys.exit(f"{MANIFEST} introuvable. Lance d'abord download_corpus.py")
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def process_document(row: dict, size: int, overlap: int, out) -> dict | None:
    """Extrait, nettoie et découpe un PDF, puis écrit ses passages.

    Pour chaque page du PDF :
      1. le texte est extrait avec PyMuPDF puis nettoyé (``clean_text``) ;
      2. si la page contient moins de ``MIN_PAGE_CHARS`` caractères, elle
         est comptée comme vide et ignorée (page blanche, image, scan) ;
      3. sinon, le texte est découpé (``chunk_text``) et chaque passage est
         écrit sur ``out`` sous la forme d'une ligne JSON.

    Le découpage se fait page par page : un passage ne s'étend jamais sur
    deux pages, ce qui permet de citer « document, page N ».

    Champs de chaque ligne JSON :
      chunk_id     : identifiant unique ``<doc_id>_p<page sur 4 chiffres>_c<index sur 2 chiffres>``
      doc_id, company, doc_type, year, language, filename : repris du manifeste
      doc_sha256   : empreinte du PDF source
      page         : numéro de page, à partir de 1
      chunk_index  : position du passage dans la page, à partir de 0
      char_count   : longueur du passage
      text         : contenu du passage

    Args:
        row: une ligne du manifeste (voir ``read_manifest``).
        size: taille maximale d'un passage (transmise à ``chunk_text``).
        overlap: chevauchement entre passages (transmis à ``chunk_text``).
        out: fichier texte déjà ouvert en écriture (sortie JSONL).

    Returns:
        ``None`` si le PDF est absent de data/raw/ (un message est affiché).
        Sinon, des statistiques ``{"id", "pages", "empty_pages", "chunks",
        "chars"}`` : nombre de pages, de pages vides, de passages écrits et
        total des caractères des passages.

    Raises:
        Les erreurs de PyMuPDF si le PDF est corrompu ou ne peut pas être ouvert.
    """
    pdf_path = RAW_DIR / row["filename"]
    if not pdf_path.exists():
        print(f"[ignoré]  {row['id']} : fichier absent ({pdf_path})")
        return None

    doc_hash = sha256_of(pdf_path)
    stats = {"id": row["id"], "pages": 0, "empty_pages": 0, "chunks": 0, "chars": 0}

    with fitz.open(pdf_path) as pdf:
        stats["pages"] = pdf.page_count
        for page_index in range(pdf.page_count):
            text = clean_text(pdf[page_index].get_text("text"))
            if len(text) < MIN_PAGE_CHARS:
                stats["empty_pages"] += 1
                continue
            for chunk_index, chunk in enumerate(chunk_text(text, size, overlap)):
                page_number = page_index + 1
                record = {
                    "chunk_id": f"{row['id']}_p{page_number:04d}_c{chunk_index:02d}",
                    "doc_id": row["id"],
                    "company": row["company"],
                    "doc_type": row["doc_type"],
                    "year": row["year"],
                    "language": row["language"],
                    "filename": row["filename"],
                    "doc_sha256": doc_hash,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk),
                    "text": chunk,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["chunks"] += 1
                stats["chars"] += len(chunk)
    return stats


def main() -> int:
    """Point d'entrée : extrait tout le corpus vers data/processed/chunks.jsonl.

    Étapes :
      1. lit les options ``--chunk-size`` (1200 par défaut) et ``--overlap``
         (150 par défaut) et vérifie que overlap < chunk-size ;
      2. lit le manifeste et crée data/processed/ si nécessaire ;
      3. réécrit entièrement chunks.jsonl en appelant ``process_document``
         pour chaque document ;
      4. affiche un tableau récapitulatif par document (pages, pages vides,
         passages, longueur moyenne). Un avertissement apparaît quand plus de
         30 % des pages sont vides, signe probable d'un PDF scanné qui
         nécessite de l'OCR.

    Returns:
        Code de sortie : 0. Le programme s'arrête avec un message d'erreur
        si les options sont incohérentes ou si le manifeste est absent.
    """
    parser = argparse.ArgumentParser(description="Extraction et découpage des PDF")
    parser.add_argument("--chunk-size", type=int, default=1200, help="taille max d'un passage (caractères)")
    parser.add_argument("--overlap", type=int, default=150, help="chevauchement entre passages (caractères)")
    args = parser.parse_args()

    if args.overlap >= args.chunk_size:
        sys.exit("--overlap doit être inférieur à --chunk-size")

    rows = read_manifest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = []
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for row in rows:
            print(f"[extraction] {row['id']} ...")
            stats = process_document(row, args.chunk_size, args.overlap, out)
            if stats:
                all_stats.append(stats)

    print("\nRésumé")
    print(f"{'document':<22}{'pages':>7}{'vides':>7}{'passages':>10}{'moy. car.':>11}")
    for s in all_stats:
        avg = s["chars"] // s["chunks"] if s["chunks"] else 0
        print(f"{s['id']:<22}{s['pages']:>7}{s['empty_pages']:>7}{s['chunks']:>10}{avg:>11}")
        if s["pages"] and s["empty_pages"] / s["pages"] > 0.3:
            print(f"  ! plus de 30 % de pages vides dans {s['id']} : PDF scanné ? (OCR nécessaire)")

    total = sum(s["chunks"] for s in all_stats)
    print(f"\n{total} passages écrits dans {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
