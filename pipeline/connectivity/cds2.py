\
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..utils import requests_post_json


def l1000cds2_payload(up_genes: List[str], dn_genes: List[str], tag: str, mode: str = "reverse", limit: int = 50) -> Dict[str, Any]:
    """
    Build payload for L1000CDS2 /query endpoint.
    mode:
      - reverse: aggravate=False
      - mimic/aggravate: aggravate=True
    """
    aggravate = (mode.lower() not in ("reverse", "rev", "inhibit", "reversal"))
    data = {
        "upGenes": [g.upper() for g in up_genes],
        "dnGenes": [g.upper() for g in dn_genes],
    }
    config = {
        "aggravate": aggravate,
        "searchMethod": "geneSet",
        "share": False,
        "combination": False,
        "db-version": "latest",
        "limit": int(limit),
    }
    meta = [{"key": "Tag", "value": tag}]
    return {"data": data, "config": config, "meta": meta}


def l1000cds2_query_with_autoshrink(
    up: List[str],
    dn: List[str],
    urls: List[str],
    tag: str,
    mode: str = "reverse",
    timeout: int = 120,
    limit: int = 50,
    tops: Optional[List[int]] = None,
    min_each: int = 10,
) -> Dict[str, Any]:
    """
    Try multiple top-N sizes and multiple URLs; attach _used_top_n and _used_url.
    """
    if tops is None:
        tops = [len(up), 250, 200, 150, 120, 100, 80, 60, 40, 30]
    # unique and descending
    tops2 = []
    for t in tops:
        if t is None:
            continue
        t = int(t)
        if t >= min_each and t not in tops2:
            tops2.append(t)

    last_err = None
    for t in tops2:
        up2 = up[:t]
        dn2 = dn[:t]
        if len(up2) < min_each or len(dn2) < min_each:
            continue
        payload = l1000cds2_payload(up2, dn2, tag=tag, mode=mode, limit=limit)
        for url in urls:
            try:
                res = requests_post_json(url, payload, timeout=timeout)
                res["_used_top_n"] = t
                res["_used_url"] = url
                return res
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"L1000CDS2 failed after autoshrink. Last error: {repr(last_err)}")
