\
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _normalize_name(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    return x


def _rank_to_score(series: pd.Series, ascending: bool) -> pd.Series:
    """
    Convert a sortable numeric series to [0,1] rank scores (higher is better).
    """
    s = pd.to_numeric(series, errors="coerce")
    n = int(s.notna().sum())
    if n == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    r = s.rank(ascending=ascending, method="min")
    # best rank -> 1.0
    return (n - r + 1) / n


def standardize_cds2_topmeta(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["drug", "cds2_score"])
    d = df.copy()
    # known fields: pert_desc, pert_id, cell_id, score
    if "pert_desc" in d.columns:
        d["drug"] = d["pert_desc"].fillna("")
    elif "pert_id" in d.columns:
        d["drug"] = d["pert_id"].fillna("")
    else:
        d["drug"] = ""
    d["drug"] = d["drug"].astype(str).str.strip()
    d["cds2_score"] = pd.to_numeric(d.get("score"), errors="coerce")
    return d[["drug", "cds2_score"]].dropna(subset=["drug"]).drop_duplicates()


def standardize_l1000fwd_hits(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["drug", "l1000fwd_score"])
    d = df.copy()
    d["drug"] = d.get("pert_desc")
    if d["drug"].isna().all():
        d["drug"] = d.get("pert_id")
    d["drug"] = d["drug"].astype(str).str.strip()
    d["l1000fwd_score"] = pd.to_numeric(d.get("score"), errors="coerce")
    return d[["drug", "l1000fwd_score"]].dropna(subset=["drug"]).drop_duplicates()


def standardize_enrichr_rev(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["drug", "enrichr_score"])
    d = df.copy()
    d["drug"] = d.get("term").astype(str).str.strip()
    d["enrichr_score"] = pd.to_numeric(d.get("enrichr_reversal_score"), errors="coerce")
    return d[["drug", "enrichr_score"]].dropna(subset=["drug"]).drop_duplicates()


def rank_compounds_for_gse(
    gse: str,
    df_cds2: pd.DataFrame,
    df_fwd: pd.DataFrame,
    df_enrichr: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 50,
) -> pd.DataFrame:
    if weights is None:
        weights = {"cds2": 1.0, "l1000fwd": 1.0, "enrichr": 0.7}

    c = standardize_cds2_topmeta(df_cds2)
    f = standardize_l1000fwd_hits(df_fwd)
    e = standardize_enrichr_rev(df_enrichr)

    # merge
    df = pd.DataFrame({"drug": pd.unique(pd.concat([c["drug"], f["drug"], e["drug"]], ignore_index=True))})
    df["drug"] = df["drug"].astype(str).str.strip()
    df = df[df["drug"] != ""].copy()

    df = df.merge(c, on="drug", how="left")
    df = df.merge(f, on="drug", how="left")
    df = df.merge(e, on="drug", how="left")

    # rank scores
    df["cds2_rank_score"] = _rank_to_score(df["cds2_score"], ascending=False)  # bigger better (heuristic)
    df["l1000fwd_rank_score"] = _rank_to_score(df["l1000fwd_score"], ascending=False)
    df["enrichr_rank_score"] = _rank_to_score(df["enrichr_score"], ascending=False)

    df["final_score"] = (
        float(weights.get("cds2", 1.0)) * df["cds2_rank_score"].fillna(0.0) +
        float(weights.get("l1000fwd", 1.0)) * df["l1000fwd_rank_score"].fillna(0.0) +
        float(weights.get("enrichr", 0.7)) * df["enrichr_rank_score"].fillna(0.0)
    )

    df.insert(0, "gse", gse)
    df = df.sort_values(["final_score", "cds2_score", "l1000fwd_score", "enrichr_score"], ascending=[False, False, False, False])
    return df.head(int(top_k)).reset_index(drop=True)


def aggregate_across_gse(df_ranked_all: pd.DataFrame) -> pd.DataFrame:
    if df_ranked_all is None or df_ranked_all.empty:
        return pd.DataFrame()

    d = df_ranked_all.copy()
    d["drug_key"] = d["drug"].astype(str).str.lower().str.strip()

    agg = (d.groupby("drug_key", as_index=False)
           .agg(
               drug=("drug", "first"),
               n_gse=("gse", "nunique"),
               gses=("gse", lambda x: "|".join(sorted(set(map(str, x))))),
               final_score_sum=("final_score", "sum"),
               final_score_mean=("final_score", "mean"),
               best_final_score=("final_score", "max"),
           )
           .sort_values(["n_gse", "final_score_sum", "best_final_score"], ascending=[False, False, False])
           )

    return agg.reset_index(drop=True)
