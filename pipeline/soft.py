\
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .ftp import gse_soft_url
from .utils import download_file


def download_family_soft(gse: str, raw_gse_dir: Path, timeout: int = 120) -> Path:
    """
    Download GSE family SOFT (.soft.gz) into raw_gse_dir.
    """
    raw_gse_dir.mkdir(parents=True, exist_ok=True)
    url = gse_soft_url(gse)
    out = raw_gse_dir / f"{gse.upper()}_family.soft.gz"
    return download_file(url, out, timeout=timeout)


def _parse_kv_line(s: str) -> Optional[Tuple[str, str]]:
    # characteristics lines can be "key: value" or plain text
    if ":" not in s:
        return None
    k, v = s.split(":", 1)
    k = k.strip().lower()
    v = v.strip()
    if not k:
        return None
    return (k, v)


def parse_gse_family_soft_gz(path: Path) -> pd.DataFrame:
    """
    Parse minimal sample metadata from GSE family soft.
    Returns DataFrame indexed by gsm with columns:
      title, source, characteristics, description, supplementary, relation, characteristics_kv
    """
    rows: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}
    chars: List[str] = []
    descs: List[str] = []
    supps: List[str] = []
    rels: List[str] = []
    char_kv: Dict[str, str] = {}

    def flush():
        nonlocal cur, chars, descs, supps, rels, char_kv
        if cur.get("gsm"):
            cur["characteristics"] = " | ".join([c for c in chars if c])
            cur["description"] = " | ".join([d for d in descs if d])
            cur["supplementary"] = " | ".join([s for s in supps if s])
            cur["relation"] = " | ".join([r for r in rels if r])
            cur["characteristics_kv"] = dict(char_kv)
            rows.append(cur)
        cur = {}
        chars, descs, supps, rels = [], [], [], []
        char_kv = {}

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE"):
                flush()
                m = re.search(r"(GSM\d+)", line)
                cur = {"gsm": m.group(1) if m else None}
                continue

            if not cur:
                continue

            # Title / source
            if line.startswith("!Sample_title"):
                cur["title"] = line.split("=", 1)[-1].strip()
            elif line.startswith("!Sample_source_name_ch1"):
                cur["source"] = line.split("=", 1)[-1].strip()

            # Characteristics
            elif line.startswith("!Sample_characteristics_ch1"):
                v = line.split("=", 1)[-1].strip()
                chars.append(v)
                kv = _parse_kv_line(v)
                if kv:
                    k, vv = kv
                    # keep first occurrence; later are often duplicates
                    if k not in char_kv:
                        char_kv[k] = vv

            # Description
            elif line.startswith("!Sample_description"):
                descs.append(line.split("=", 1)[-1].strip())

            # supplementary file
            elif line.startswith("!Sample_supplementary_file"):
                supps.append(line.split("=", 1)[-1].strip())

            # relation (BioSample/SRA)
            elif line.startswith("!Sample_relation"):
                rels.append(line.split("=", 1)[-1].strip())

            # End of sample block
            elif line.startswith("^") and not line.startswith("^SAMPLE"):
                # new section
                flush()
                cur = {}
                continue

    flush()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("gsm")
    # Ensure expected columns exist
    for c in ["title", "source", "characteristics", "description", "supplementary", "relation", "characteristics_kv"]:
        if c not in df.columns:
            df[c] = "" if c != "characteristics_kv" else [{} for _ in range(df.shape[0])]
    return df
