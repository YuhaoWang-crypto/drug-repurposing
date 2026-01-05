\
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

from .soft import download_family_soft, parse_gse_family_soft_gz
from .label import label_conditions, condition_counts
from .matrix import download_and_load_matrix, align_expr_and_meta, int_like_fraction
from .de import compute_de_bulk_ttest
from .signature import build_signature_from_de, save_signature
from .rank import rank_compounds_for_gse
from .utils import safe_mkdir, write_json

from .connectivity.cds2 import l1000cds2_query_with_autoshrink
from .connectivity.l1000fwd import l1000fwd_query
from .connectivity.enrichr import enrichr_run_dual

try:
    from .ppi import load_ppi_edges, ppi_module_metrics
except Exception:
    load_ppi_edges = None
    ppi_module_metrics = None


Logger = Callable[[str], None]


def run_one_gse(gse: str, config: Dict[str, Any], logger: Optional[Logger] = None) -> Dict[str, Any]:
    log = logger or (lambda s: None)
    gse = gse.strip().upper()

    project_dir = Path(config.get("project_dir", "./repurpose_pipeline")).expanduser().resolve()
    raw_gse_dir = safe_mkdir(project_dir / "raw" / gse)
    proc_gse_dir = safe_mkdir(project_dir / "proc" / gse)

    out: Dict[str, Any] = {"gse": gse, "project_dir": str(project_dir)}

    # 1) SOFT
    log(f"[{gse}] Download SOFT...")
    soft_path = download_family_soft(gse, raw_gse_dir, timeout=180)
    out["soft_path"] = str(soft_path)

    log(f"[{gse}] Parse SOFT...")
    df_meta = parse_gse_family_soft_gz(soft_path)
    if df_meta.empty:
        raise RuntimeError(f"[{gse}] Parsed SOFT meta empty.")
    df_meta.to_csv(proc_gse_dir / "sample_meta.csv")
    out["sample_meta_csv"] = str(proc_gse_dir / "sample_meta.csv")

    # 2) Condition labeling
    log(f"[{gse}] Infer conditions...")
    df_meta_l = label_conditions(df_meta, gse=gse, config=config)
    df_meta_l.to_csv(proc_gse_dir / "sample_meta_labeled.csv")
    out["sample_meta_labeled_csv"] = str(proc_gse_dir / "sample_meta_labeled.csv")

    counts = condition_counts(df_meta_l)
    out["condition_counts"] = counts
    log(f"[{gse}] Condition counts: {counts}")

    # 3) Download & load matrix (bulk)
    log(f"[{gse}] Download & load expression matrix (bulk)...")
    mode, matrix_path, df_expr = download_and_load_matrix(gse, raw_gse_dir, timeout=240)
    out["matrix_mode"] = mode
    out["matrix_path"] = str(matrix_path)
    log(f"[{gse}] Loaded matrix shape = {df_expr.shape}")

    # 4) Align columns with meta
    df_expr_al, meta_al = align_expr_and_meta(df_expr, df_meta_l, gse=gse, config=config)
    out["aligned_samples"] = int(df_expr_al.shape[1])
    log(f"[{gse}] Aligned samples = {df_expr_al.shape[1]}")

    # 5) DE
    eps = float(config.get("de_eps", 1e-6))
    log(f"[{gse}] Differential expression (ttest)...")
    df_de = compute_de_bulk_ttest(df_expr_al, meta_al, eps=eps)
    de_csv = proc_gse_dir / "DE_bulk_case_vs_control.csv"
    df_de.to_csv(de_csv, index=False)
    out["de_csv"] = str(de_csv)

    # 6) Signature
    top_n = int(config.get("signature_top_n", 150))
    min_abs = float(config.get("sig_min_abs_log2fc", 0.25))
    sig = build_signature_from_de(df_de, top_n=top_n, min_abs_log2fc=min_abs, species="human")
    sig_dir = safe_mkdir(proc_gse_dir / "signature")
    up_p, dn_p = save_signature(sig, sig_dir)
    out["signature_up"] = str(up_p)
    out["signature_dn"] = str(dn_p)
    log(f"[{gse}] Signature: UP={len(sig['up'])} DN={len(sig['dn'])}")

    # 7) Connectivity
    df_cds2 = pd.DataFrame()
    df_fwd = pd.DataFrame()
    df_enrichr_long = pd.DataFrame()
    df_enrichr_rev = pd.DataFrame()

    # L1000CDS2
    if bool(config.get("run_l1000cds2", True)):
        urls = list(config.get("l1000_urls") or [])
        mode = str(config.get("l1000_mode", "reverse"))
        limit = int(config.get("l1000_limit", 50))
        timeout = int(config.get("l1000_timeout", 120))
        log(f"[{gse}] Query L1000CDS2...")
        res = l1000cds2_query_with_autoshrink(sig["up"], sig["dn"], urls=urls, tag=f"{gse} {mode}", mode=mode, timeout=timeout, limit=limit)
        (proc_gse_dir / "l1000cds2_raw.json").write_text(json.dumps(res, indent=2))
        out["l1000cds2_raw_json"] = str(proc_gse_dir / "l1000cds2_raw.json")
        df_cds2 = pd.DataFrame(res.get("topMeta", []) or [])
        df_cds2.to_csv(proc_gse_dir / "l1000cds2_topMeta.csv", index=False)
        out["l1000cds2_top_csv"] = str(proc_gse_dir / "l1000cds2_topMeta.csv")
        log(f"[{gse}] L1000CDS2 hits: {df_cds2.shape[0]}")

    # L1000FWD
    if bool(config.get("run_l1000fwd", True)):
        bases = list(config.get("l1000fwd_base_urls") or [])
        timeout = int(config.get("l1000fwd_timeout", 120))
        limit = int(config.get("l1000fwd_limit", 50))
        log(f"[{gse}] Query L1000FWD (opposite)...")
        try:
            df_fwd = l1000fwd_query(sig["up"], sig["dn"], base_urls=bases, timeout=timeout, limit=limit, direction="opposite")
            df_fwd.to_csv(proc_gse_dir / "l1000fwd_opposite.csv", index=False)
            out["l1000fwd_csv"] = str(proc_gse_dir / "l1000fwd_opposite.csv")
            log(f"[{gse}] L1000FWD hits: {df_fwd.shape[0]}")
        except Exception as e:
            out["l1000fwd_error"] = repr(e)
            log(f"[{gse}] [WARN] L1000FWD failed: {repr(e)}")

    # Enrichr
    if bool(config.get("run_enrichr", True)):
        base = str(config.get("enrichr_base_url", "https://maayanlab.cloud/Enrichr"))
        libs = list(config.get("enrichr_libraries") or [])
        log(f"[{gse}] Query Enrichr libraries ({len(libs)}) ...")
        try:
            df_long, df_rev = enrichr_run_dual(base, sig["up"], sig["dn"], libraries=libs, desc_prefix=gse, top_k=100)
            df_long.to_csv(proc_gse_dir / "enrichr_long.csv", index=False)
            df_rev.to_csv(proc_gse_dir / "enrichr_reversal_rank.csv", index=False)
            out["enrichr_long_csv"] = str(proc_gse_dir / "enrichr_long.csv")
            out["enrichr_rev_csv"] = str(proc_gse_dir / "enrichr_reversal_rank.csv")
            df_enrichr_long, df_enrichr_rev = df_long, df_rev
            log(f"[{gse}] Enrichr reversal hits: {df_rev.shape[0]}")
        except Exception as e:
            out["enrichr_error"] = repr(e)
            log(f"[{gse}] [WARN] Enrichr failed: {repr(e)}")

    # 8) Ranking
    weights = config.get("rank_weights") or {"cds2": 1.0, "l1000fwd": 1.0, "enrichr": 0.7}
    df_rank = rank_compounds_for_gse(gse, df_cds2=df_cds2, df_fwd=df_fwd, df_enrichr=df_enrichr_rev, weights=weights, top_k=50)
    ranked_csv = proc_gse_dir / "compound_ranked.csv"
    df_rank.to_csv(ranked_csv, index=False)
    out["compound_ranked_csv"] = str(ranked_csv)
    log(f"[{gse}] Ranked compounds saved: {ranked_csv} rows={df_rank.shape[0]}")

    # 9) Optional PPI module metrics (global metrics; not drug-specific)
    ppi_csv = str(config.get("ppi_edges_csv") or "").strip()
    if ppi_csv and load_ppi_edges is not None and ppi_module_metrics is not None:
        try:
            G = load_ppi_edges(Path(ppi_csv))
            module_genes = set(config.get('module_gene_symbols') or [])
            metrics = ppi_module_metrics(G, module_genes=module_genes, sig_up=set(sig["up"]), sig_dn=set(sig["dn"]), max_dist=int(config.get("ppi_max_dist", 2)))
            write_json(proc_gse_dir / "ppi_module_metrics.json", metrics)
            out["ppi_module_metrics_json"] = str(proc_gse_dir / "ppi_module_metrics.json")
        except Exception as e:
            out["ppi_error"] = repr(e)
            log(f"[{gse}] [WARN] PPI metrics failed: {repr(e)}")

    return out
