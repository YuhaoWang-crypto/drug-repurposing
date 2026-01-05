\
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from ..utils import requests_get, requests_post_json


_BRDCODE_RE = re.compile(r"(BRD-[A-Z0-9]+)", re.I)


def _norm_base(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    # accept '.../l1000fwd' or '.../l1000fwd/'
    url = url.rstrip("/")
    # Some users paste the /api_page; remove if present
    url = url.replace("/api_page", "")
    return url + "/"


def extract_pert_id_from_sig_id(sig_id: str) -> Optional[str]:
    m = _BRDCODE_RE.search(sig_id or "")
    if m:
        return m.group(1).upper()
    return None


def extract_cell_id_from_sig_id(sig_id: str) -> Optional[str]:
    # Example: CPC006_HA1E_24H:BRD-K08417745:88.8 -> cell_id is HA1E
    # Another: CPC006_HA1E_24H:BRD-A70155556-001-04-4:40
    try:
        left = sig_id.split(":", 1)[0]
        parts = left.split("_")
        if len(parts) >= 2:
            return parts[1]
    except Exception:
        pass
    return None


@lru_cache(maxsize=2048)
def l1000fwd_synonyms(base_url: str, query: str, timeout: int = 60) -> List[Dict[str, Any]]:
    """
    GET /synonyms/<query_string>
    """
    base = _norm_base(base_url)
    url = base + "synonyms/" + requests.utils.quote(query)
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        return []
    try:
        return r.json()
    except Exception:
        return []


def l1000fwd_sig_search(base_url: str, up: List[str], dn: List[str], timeout: int = 120) -> Optional[str]:
    """
    POST /sig_search -> {"result_id": "..."}
    """
    base = _norm_base(base_url)
    url = base + "sig_search"
    payload = {"up_genes": [g.upper() for g in up], "down_genes": [g.upper() for g in dn]}
    try:
        res = requests_post_json(url, payload, timeout=timeout)
        rid = res.get("result_id")
        if isinstance(rid, str) and rid:
            return rid
    except Exception:
        return None
    return None


def l1000fwd_topn(base_url: str, result_id: str, timeout: int = 120) -> Dict[str, Any]:
    """
    GET /result/topn/<result_id>
    """
    base = _norm_base(base_url)
    url = base + "result/topn/" + requests.utils.quote(result_id)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def l1000fwd_query(
    up: List[str],
    dn: List[str],
    base_urls: List[str],
    timeout: int = 120,
    limit: int = 50,
    direction: str = "opposite",
) -> pd.DataFrame:
    """
    Run L1000FWD query and return DataFrame of hits.
    direction: "opposite" (reversal) or "similar" (mimic)
    """
    last_err = None
    for bu in base_urls:
        base = _norm_base(bu)
        try:
            rid = l1000fwd_sig_search(base, up, dn, timeout=timeout)
            if not rid:
                raise RuntimeError("No result_id returned")
            js = l1000fwd_topn(base, rid, timeout=timeout)
            hits = js.get(direction) or js.get(direction.lower()) or []
            if not isinstance(hits, list):
                hits = []
            rows = []
            for h in hits[: int(limit)]:
                sig_id = h.get("sig_id") or h.get("sig") or h.get("id")
                score = h.get("score") or h.get("similarity") or h.get("dist")
                if not sig_id:
                    continue
                pert_id = extract_pert_id_from_sig_id(str(sig_id))
                cell_id = extract_cell_id_from_sig_id(str(sig_id))
                pert_desc = None
                if pert_id:
                    syn = l1000fwd_synonyms(base, pert_id)
                    if syn:
                        # take first
                        pert_desc = syn[0].get("Name") or syn[0].get("name")
                rows.append({
                    "sig_id": sig_id,
                    "pert_id": pert_id,
                    "pert_desc": pert_desc,
                    "cell_id": cell_id,
                    "score": score,
                    "direction": direction,
                    "_used_url": base,
                    "_result_id": rid,
                })
            return pd.DataFrame(rows)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"L1000FWD query failed on all base_urls. Last error: {repr(last_err)}")
