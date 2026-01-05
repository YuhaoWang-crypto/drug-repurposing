\
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


def compute_de_bulk_ttest(
    df_expr: pd.DataFrame,
    meta: pd.DataFrame,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Compute log2FC and Welch t-test on log2(expr+eps).
    meta must have a 'condition' column with 'case' and 'control'.

    Returns: DataFrame with columns: gene_raw, log2fc, pval, padj
    """
    # Ensure numeric
    X = df_expr.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=0, how="all").dropna(axis=1, how="all")

    cond = meta["condition"].astype(str)
    case_cols = [c for c in X.columns if cond.get(c, "unknown") == "case"]
    ctrl_cols = [c for c in X.columns if cond.get(c, "unknown") == "control"]

    if len(case_cols) == 0 or len(ctrl_cols) == 0:
        raise RuntimeError(f"Not enough samples for DE: case={len(case_cols)} control={len(ctrl_cols)}")

    X_case = np.log2(X[case_cols] + eps)
    X_ctrl = np.log2(X[ctrl_cols] + eps)

    # Means
    mean_case = X_case.mean(axis=1)
    mean_ctrl = X_ctrl.mean(axis=1)
    log2fc = mean_case - mean_ctrl

    # t-test
    t, p = ttest_ind(X_case.T, X_ctrl.T, equal_var=False, nan_policy="omit")
    p = np.asarray(p, dtype=float)

    # FDR
    padj = multipletests(p, method="fdr_bh")[1]

    out = pd.DataFrame({
        "gene_raw": X.index.astype(str),
        "log2fc": log2fc.values,
        "pval": p,
        "padj": padj,
    }).sort_values("padj", ascending=True)
    return out
