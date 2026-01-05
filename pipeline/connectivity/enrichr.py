\
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


def _norm_base(url: str) -> str:
    url = url.strip().rstrip("/")
    return url


def enrichr_add_list(base_url: str, genes: List[str], description: str = "", timeout: int = 60) -> int:
    """
    POST /addList

    Enrichr expects:
      - 'list': newline-separated gene symbols
      - 'description': free text
    """
    base = _norm_base(base_url)
    url = base + "/addList"

    gene_str = "\n".join([g.strip().upper() for g in genes if g and str(g).strip()]) + "\n"
    data = {"list": gene_str, "description": description}

    r = requests.post(url, data=data, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    uid = js.get("userListId")
    if uid is None:
        raise RuntimeError(f"Enrichr addList returned no userListId: {js}")
    return int(uid)


def enrichr_enrich(base_url: str, user_list_id: int, library: str, timeout: int = 120) -> pd.DataFrame:
    """
    GET /enrich?userListId=...&backgroundType=...
    """
    base = _norm_base(base_url)
    url = base + "/enrich"
    params = {"userListId": int(user_list_id), "backgroundType": library}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    res = js.get(library) or js.get("results") or []
    rows = []
    for item in res:
        # [rank, term, pval, z, combined, overlap, adj_p, old_p, old_adj]
        if not isinstance(item, list) or len(item) < 7:
            continue
        rows.append({
            "rank": item[0],
            "term": item[1],
            "pval": item[2],
            "zscore": item[3],
            "combined_score": item[4],
            "overlap": item[5],
            "adj_p": item[6],
            "library": library,
        })
    return pd.DataFrame(rows)


def enrichr_run_dual(
    base_url: str,
    up: List[str],
    dn: List[str],
    libraries: List[str],
    desc_prefix: str = "",
    timeout_add: int = 60,
    timeout_enrich: int = 120,
    top_k: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Enrichr for UP and DN lists across libraries.
    Returns:
      df_long: all results with query label
      df_reversal: aggregated reversal-like score per term
    """
    uid_up = enrichr_add_list(base_url, up, description=f"{desc_prefix} UP", timeout=timeout_add)
    uid_dn = enrichr_add_list(base_url, dn, description=f"{desc_prefix} DN", timeout=timeout_add)

    all_rows = []
    for lib in libraries:
        df_u = enrichr_enrich(base_url, uid_up, lib, timeout=timeout_enrich).head(top_k)
        df_u["query"] = "UP"
        df_d = enrichr_enrich(base_url, uid_dn, lib, timeout=timeout_enrich).head(top_k)
        df_d["query"] = "DN"
        all_rows.append(df_u)
        all_rows.append(df_d)

    df_long = pd.concat([d for d in all_rows if d is not None and not d.empty], ignore_index=True) if all_rows else pd.DataFrame()

    if df_long.empty:
        return df_long, pd.DataFrame()

    def neglog10(x):
        try:
            x = float(x)
            if x <= 0:
                return 50.0
            return -math.log10(x)
        except Exception:
            return 0.0

    df_long["neglog10_adj_p"] = df_long["adj_p"].apply(neglog10)

    # For reversal:
    #   - drug *_down libraries should overlap your UP genes
    #   - drug *_up libraries should overlap your DN genes
    rev_mask = (
        (df_long["library"].str.lower().str.endswith("_down") & (df_long["query"] == "UP")) |
        (df_long["library"].str.lower().str.endswith("_up") & (df_long["query"] == "DN"))
    )
    df_rev = df_long[rev_mask].copy()
    if df_rev.empty:
        return df_long, pd.DataFrame()

    agg = (df_rev
           .groupby("term", as_index=False)
           .agg(
               enrichr_reversal_score=("neglog10_adj_p", "sum"),
               enrichr_hits=("term", "size"),
               best_adj_p=("adj_p", "min"),
               best_combined=("combined_score", "max"),
           )
           .sort_values(["enrichr_reversal_score", "best_combined"], ascending=[False, False])
           )

    return df_long, agg
