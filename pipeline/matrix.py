\
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .ftp import gse_ftp_base, list_suppl_files, pick_bulk_matrix_file, pick_10x_triplet
from .utils import download_file


def download_suppl_file(gse: str, filename: str, raw_gse_dir: Path, timeout: int = 180) -> Path:
    url = f"{gse_ftp_base(gse)}/suppl/{filename}"
    out = raw_gse_dir / filename
    return download_file(url, out, timeout=timeout)


def guess_sep_from_header(line: str) -> str:
    if line.count("\t") >= line.count(","):
        return "\t"
    return ","


def load_bulk_matrix(path: Path) -> pd.DataFrame:
    """
    Load TSV/CSV/TXT(.gz) into DataFrame with genes as index and samples as columns.
    Best-effort; GEO matrices are not standardized.
    """
    # probe first line to guess separator
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            header = f.readline()
    else:
        with open(path, "rt", encoding="utf-8", errors="replace") as f:
            header = f.readline()

    sep = guess_sep_from_header(header)

    df = pd.read_csv(path, sep=sep, engine="python")
    if df.shape[1] < 2:
        raise RuntimeError(f"Matrix file has <2 columns: {path}")

    # If gene column exists
    gene_col_candidates = [c for c in df.columns if str(c).lower() in ("gene", "gene_id", "geneid", "symbol", "gene_symbol", "genes", "ensembl", "id")]
    if gene_col_candidates:
        gene_col = gene_col_candidates[0]
        df = df.set_index(gene_col)
    else:
        # assume first column is gene id
        df = df.set_index(df.columns[0])

    # Coerce numeric columns
    for c in list(df.columns):
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # ensure unique gene ids
    df.index = df.index.astype(str)
    df = df[~df.index.duplicated(keep="first")]
    return df


def int_like_fraction(df: pd.DataFrame, n_probe: int = 5000) -> float:
    """
    fraction of values that are near integers.
    """
    x = df.to_numpy().ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    if x.size > n_probe:
        x = np.random.choice(x, size=n_probe, replace=False)
    frac = float(np.mean(np.isclose(x, np.round(x), atol=1e-6)))
    return frac


def map_columns_to_gsm_by_text(df_expr: pd.DataFrame, df_meta: pd.DataFrame) -> Dict[str, str]:
    """
    Attempt to map expression matrix columns to GSM IDs.
    Returns mapping {original_col_name -> gsm}
    """
    cols = list(df_expr.columns.astype(str))
    mapping: Dict[str, str] = {}

    gsm_set = set(df_meta.index.astype(str))

    # 1) direct match
    for c in cols:
        if c in gsm_set:
            mapping[c] = c

    # 2) GSM embedded in column name
    for c in cols:
        if c in mapping:
            continue
        m = re.search(r"(GSM\d+)", c, flags=re.I)
        if m and m.group(1) in gsm_set:
            mapping[c] = m.group(1)

    # 3) match by title tokens (very crude)
    if len(mapping) < max(2, int(0.5 * len(cols))):
        title_map = df_meta["title"].to_dict() if "title" in df_meta.columns else {}
        norm = lambda s: re.sub(r"[^a-z0-9]+", "", str(s).lower())

        col_norm = {c: norm(c) for c in cols}
        gsm_norm = {gsm: norm(title_map.get(gsm, "")) for gsm in gsm_set}

        for c in cols:
            if c in mapping:
                continue
            cn = col_norm.get(c, "")
            if not cn:
                continue
            best = None
            best_score = 0
            for gsm, tn in gsm_norm.items():
                if not tn:
                    continue
                score = 0
                if cn in tn or tn in cn:
                    score = max(len(cn), len(tn))
                else:
                    for k in range(min(len(cn), len(tn)), 4, -1):
                        if cn[:k] in tn or tn[:k] in cn:
                            score = k
                            break
                if score > best_score:
                    best_score = score
                    best = gsm
            if best is not None and best_score >= 6:
                mapping[c] = best

    return mapping


def infer_condition_from_colname(col: str, gse: str, config: Dict[str, Any]) -> str:
    """
    If we cannot map to GSM, infer case/control from column names using override regex.
    """
    ov = (config.get("per_gse_overrides") or {}).get(gse.upper(), {}) or {}
    col_case_re = ov.get("col_case_regex")
    col_ctrl_re = ov.get("col_control_regex")

    if col_case_re:
        try:
            if re.search(col_case_re, col):
                return "case"
        except re.error:
            pass
    if col_ctrl_re:
        try:
            if re.search(col_ctrl_re, col):
                return "control"
        except re.error:
            pass

    cl = col.lower()
    if re.search(r"(ko|knockout|mut|mutation|patient|case|disease)", cl):
        return "case"
    if re.search(r"(wt|control|ctrl|healthy|normal|vehicle)", cl):
        return "control"
    return "unknown"


def align_expr_and_meta(
    df_expr: pd.DataFrame,
    df_meta_labeled: pd.DataFrame,
    gse: str,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Try mapping columns -> GSM, then align meta.
    If mapping is partial, keep unmapped columns and label them via column-name heuristics.

    Returns (df_expr_aligned, df_meta_aligned) where meta index matches expr columns
    and meta has a 'condition' column.
    """
    mapping = map_columns_to_gsm_by_text(df_expr, df_meta_labeled)
    mapped_cols = [c for c in df_expr.columns if str(c) in mapping]

    if len(mapped_cols) >= 2:
        df2 = df_expr.copy()
        df2.columns = [mapping.get(str(c), str(c)) for c in df2.columns]

        meta2 = df_meta_labeled.reindex(df2.columns).copy()
        if "condition" not in meta2.columns:
            meta2["condition"] = "unknown"
        if "title" not in meta2.columns:
            meta2["title"] = meta2.index.astype(str)

        for c in meta2.index.astype(str):
            cond = str(meta2.at[c, "condition"]) if c in meta2.index else "unknown"
            if cond.lower() not in ("case", "control"):
                meta2.at[c, "condition"] = infer_condition_from_colname(c, gse=gse, config=config)
            if pd.isna(meta2.at[c, "title"]):
                meta2.at[c, "title"] = c

        return df2, meta2

    # Fallback: infer condition purely from column names
    pseudo = pd.DataFrame(index=df_expr.columns.astype(str))
    pseudo["title"] = pseudo.index
    pseudo["condition"] = [infer_condition_from_colname(c, gse=gse, config=config) for c in pseudo.index]
    return df_expr.copy(), pseudo


def download_and_load_matrix(gse: str, raw_gse_dir: Path, timeout: int = 180) -> Tuple[str, Path, pd.DataFrame]:
    """
    Download candidate matrix from suppl, load into df.
    Returns (mode, filepath, df_expr)
    mode: "bulk" or "10x"
    """
    files = list_suppl_files(gse)
    triplet = pick_10x_triplet(files)
    if triplet:
        # download triplet
        mtx, bar, feat = triplet
        download_suppl_file(gse, mtx, raw_gse_dir, timeout=timeout)
        download_suppl_file(gse, bar, raw_gse_dir, timeout=timeout)
        download_suppl_file(gse, feat, raw_gse_dir, timeout=timeout)
        raise NotImplementedError("10x triplet detected but not implemented in this Streamlit template.")

    bulk = pick_bulk_matrix_file(files)
    if not bulk:
        raise RuntimeError("No candidate bulk matrix file found in GEO suppl directory.")
    fpath = download_suppl_file(gse, bulk, raw_gse_dir, timeout=timeout)
    df = load_bulk_matrix(fpath)
    return "bulk", fpath, df
