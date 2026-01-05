\
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .utils import requests_get


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def geo_esearch(term: str, retmax: int = 40) -> List[str]:
    """
    Search GEO DataSets (GDS) database via eutils, return list of internal IDs.
    """
    params = {
        "db": "gds",
        "term": term,
        "retmode": "json",
        "retmax": int(retmax),
    }
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    ids = js.get("esearchresult", {}).get("idlist", []) or []
    return [str(x) for x in ids]


def geo_esummary(ids: List[str]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    params = {
        "db": "gds",
        "id": ",".join(ids),
        "retmode": "json",
    }
    r = requests.get(f"{EUTILS}/esummary.fcgi", params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    result = js.get("result", {})
    out = []
    for k, v in result.items():
        if k == "uids":
            continue
        if isinstance(v, dict):
            out.append(v)
    return out


def extract_gse_accession(accession: str) -> Optional[str]:
    """
    accession can be like "GSE216834" or "GDSxxxx"; we only keep GSE.
    """
    if not accession:
        return None
    m = re.search(r"(GSE\d+)", accession.upper())
    if m:
        return m.group(1)
    return None


def geo_search_candidates(queries: List[str], retmax_each: int = 40) -> pd.DataFrame:
    """
    Run multiple queries, return a de-duplicated table of GSE hits.
    """
    rows = []
    for q in queries:
        ids = geo_esearch(q, retmax=retmax_each)
        summ = geo_esummary(ids)
        for v in summ:
            acc = extract_gse_accession(str(v.get("accession", "")))
            if not acc:
                continue
            rows.append({
                "accession": acc,
                "title": v.get("title", ""),
                "summary": v.get("summary", ""),
                "gdsType": v.get("gdstype", ""),
                "n_samples": v.get("n_samples", ""),
                "PDAT": v.get("pdat", ""),
                "taxon": v.get("taxon", ""),
                "entryType": v.get("entrytype", ""),
                "query": q,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["accession"]).sort_values("PDAT", ascending=False)
    return df
