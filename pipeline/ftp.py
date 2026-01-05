\
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from .utils import parse_href_list_from_html, requests_get


def gse_num(gse: str) -> int:
    gse = gse.strip().upper()
    if not gse.startswith("GSE"):
        raise ValueError(f"Not a GSE accession: {gse}")
    return int(re.sub(r"\D+", "", gse))


def gse_ftp_base(gse: str) -> str:
    """
    Correct NCBI GEO FTP layout:
      https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{prefix}nnn/{GSE}/
    where prefix = floor(GSE / 1000).
    Example:
      GSE216834 -> prefix 216 -> .../GSE216nnn/GSE216834/
    """
    n = gse_num(gse)
    prefix = n // 1000
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{prefix}nnn/{gse.upper()}"


def gse_soft_url(gse: str) -> str:
    return f"{gse_ftp_base(gse)}/soft/{gse.upper()}_family.soft.gz"


def list_suppl_files(gse: str, timeout: int = 60) -> List[str]:
    """
    List files in the 'suppl' directory.
    Returns filenames (not full URLs).
    """
    url = f"{gse_ftp_base(gse)}/suppl/"
    r = requests_get(url, timeout=timeout)
    hrefs = parse_href_list_from_html(r.text)
    # Keep only downloadable files (not dirs)
    files = [h for h in hrefs if not h.endswith("/") and not h.startswith("?")]
    # Filter out md5
    files = [f for f in files if not f.lower().endswith(".md5")]
    return sorted(set(files))


def pick_bulk_matrix_file(files: List[str]) -> Optional[str]:
    """
    Heuristic: pick a bulk matrix-like file from GEO suppl filenames.
    """
    if not files:
        return None

    # Prioritize obvious matrices / counts
    patterns = [
        r"(matrix|count|counts|txi|cpm|tmm|fpkm|tpm|rpkm|expression|expr)",
    ]
    candidates = []
    for f in files:
        fl = f.lower()
        if any(re.search(p, fl) for p in patterns) and re.search(r"\.(txt|tsv|csv)(\.gz)?$", fl):
            candidates.append(f)

    # Exclude small annotation-only tables
    bad = re.compile(r"(metadata|meta|annotation|annot|readme|sdrf|idf|design)", re.I)
    candidates = [f for f in candidates if not bad.search(f)]

    if candidates:
        # Prefer txi/counts first
        key = lambda x: (
            0 if "txi" in x.lower() else
            1 if "count" in x.lower() else
            2 if "cpm" in x.lower() or "tmm" in x.lower() else
            3
        )
        candidates.sort(key=key)
        return candidates[0]

    # Fallback: any txt/tsv/csv.gz file
    anytab = [f for f in files if re.search(r"\.(txt|tsv|csv)(\.gz)?$", f.lower())]
    return anytab[0] if anytab else None


def pick_10x_triplet(files: List[str]) -> Optional[Tuple[str, str, str]]:
    """
    Detect 10x triplet in suppl: matrix.mtx(.gz), barcodes.tsv(.gz), features.tsv(.gz)/genes.tsv(.gz)
    Returns filenames tuple if found, else None.
    """
    fl = [f.lower() for f in files]
    def find_one(regex: str) -> Optional[str]:
        for f in files:
            if re.search(regex, f, flags=re.I):
                return f
        return None

    mtx = find_one(r"(matrix\.mtx)(\.gz)?$")
    bar = find_one(r"(barcodes\.tsv)(\.gz)?$")
    feat = find_one(r"(features\.tsv|genes\.tsv)(\.gz)?$")
    if mtx and bar and feat:
        return (mtx, bar, feat)
    return None
