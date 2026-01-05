\
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from pipeline.config import default_config
from pipeline.geo import geo_search_candidates
from pipeline.soft import download_family_soft, parse_gse_family_soft_gz
from pipeline.label import label_conditions, condition_counts, pass_flags
from pipeline.pipeline import run_one_gse
from pipeline.rank import aggregate_across_gse


st.set_page_config(page_title="CLCN GEO → Connectivity → Ranked Drugs", layout="wide")


def _init_state():
    if "config" not in st.session_state:
        st.session_state["config"] = default_config()
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    if "candidates_df" not in st.session_state:
        st.session_state["candidates_df"] = pd.DataFrame()
    if "validation_df" not in st.session_state:
        st.session_state["validation_df"] = pd.DataFrame()
    if "selected_gse" not in st.session_state:
        st.session_state["selected_gse"] = []
    if "run_results" not in st.session_state:
        st.session_state["run_results"] = []
    if "ranked_all_df" not in st.session_state:
        st.session_state["ranked_all_df"] = pd.DataFrame()
    if "aggregated_df" not in st.session_state:
        st.session_state["aggregated_df"] = pd.DataFrame()


def add_log(msg: str):
    st.session_state["logs"].append(str(msg))


def render_logs(max_lines: int = 300):
    lines = st.session_state.get("logs", [])[-max_lines:]
    st.code("\n".join(lines), language="text")


def build_geo_queries(genes: str, phenotype: str, exclude_ad: bool, extra_terms: str = "") -> List[str]:
    genes = genes.strip()
    phenotype = phenotype.strip()
    extra_terms = extra_terms.strip()

    # OR-join genes
    gene_terms = []
    for g in [x.strip() for x in genes.replace(";", ",").split(",") if x.strip()]:
        if " " in g:
            gene_terms.append(f'"{g}"')
        else:
            gene_terms.append(g)
    gene_block = " OR ".join(gene_terms) if gene_terms else ""

    pert_block = '(mutation OR mutant OR variant OR knockout OR knockdown OR CRISPR OR siRNA OR shRNA OR overexpression OR patient)'
    assay_block = '("Expression profiling by high throughput sequencing" OR RNA-seq OR "single cell" OR "single nucleus" OR scRNAseq OR snRNAseq)'

    q = ""
    if gene_block:
        q += f"({gene_block})"
    if phenotype:
        q += f" AND ({phenotype})" if q else f"({phenotype})"
    q += f" AND {pert_block}" if q else pert_block
    q += f" AND {assay_block}"

    if extra_terms:
        q += f" AND ({extra_terms})"

    if exclude_ad:
        q += ' NOT (Alzheimer OR dementia OR "AD brain")'

    return [q]


@st.cache_data(show_spinner=False)
def cached_geo_search(queries: List[str], retmax_each: int) -> pd.DataFrame:
    return geo_search_candidates(queries, retmax_each=retmax_each)


@st.cache_data(show_spinner=False)
def cached_soft_parse(gse: str, project_dir: str) -> pd.DataFrame:
    project = Path(project_dir).expanduser().resolve()
    raw_dir = project / "raw" / gse
    soft_path = download_family_soft(gse, raw_dir, timeout=180)
    df = parse_gse_family_soft_gz(soft_path)
    return df


_init_state()

st.title("CLCN GEO → Connectivity (L1000CDS2 / L1000FWD / Enrichr) → (optional) PPI → Ranked Drugs")

with st.sidebar:
    st.header("Inputs")

    genes_in = st.text_input("Target gene(s) / protein(s) (comma separated)", value="CLCN2")
    phenotype_in = st.text_input("Phenotype keywords (optional)", value="lysosome OR endosome OR chloride OR acidification")
    exclude_ad = st.checkbox("Exclude Alzheimer/dementia terms", value=True)
    extra_terms = st.text_input("Extra GEO terms (optional)", value="")

    st.divider()
    st.subheader("Config (editable JSON)")
    cfg_txt = st.text_area("Config JSON", value=json.dumps(st.session_state["config"], ensure_ascii=False, indent=2), height=380)
    if st.button("Apply config JSON"):
        try:
            st.session_state["config"] = json.loads(cfg_txt)
            st.success("Config updated.")
        except Exception as e:
            st.error(f"Invalid JSON: {repr(e)}")

    cfg = st.session_state["config"]
    project_dir = str(cfg.get("project_dir", "./repurpose_pipeline_clcn"))
    st.caption(f"project_dir = `{project_dir}`")

    st.divider()
    st.subheader("Optional PPI / module")
    # Upload PPI edges
    ppi_up = st.file_uploader("Upload PPI edges CSV (two columns a,b; gene symbols)", type=["csv"])
    if ppi_up is not None:
        try:
            project = Path(st.session_state["config"].get("project_dir", "./repurpose_pipeline_clcn")).expanduser().resolve()
            out_dir = project / "proc"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "ppi_edges_uploaded.csv"
            out_path.write_bytes(ppi_up.getvalue())
            cfg = st.session_state["config"]
            cfg["ppi_edges_csv"] = str(out_path)
            st.session_state["config"] = cfg
            st.success(f"PPI edges saved: {out_path}")
        except Exception as e:
            st.error(f"Failed to save PPI edges: {repr(e)}")

    # Module genes (symbols)
    mod_txt = st.text_area(
        "Module gene symbols (one per line; used only if you enable PPI edges)",
        value="\n".join(st.session_state["config"].get("module_gene_symbols", []) or []),
        height=140,
    )
    if st.button("Apply module genes"):
        genes = [x.strip().upper() for x in mod_txt.splitlines() if x.strip()]
        cfg = st.session_state["config"]
        cfg["module_gene_symbols"] = genes
        st.session_state["config"] = cfg
        st.success(f"Module genes updated: n={len(genes)}")

tabs = st.tabs(["1) GEO Search", "2) Validate & Select GSE", "3) Run Pipeline", "4) Results", "Logs"])

# --- Tab 1: GEO Search ---
with tabs[0]:
    st.subheader("Build query and search GEO")

    if st.button("Build & Search GEO"):
        st.session_state["logs"] = []
        add_log("[UI] Build GEO query...")

        queries = build_geo_queries(genes_in, phenotype_in, exclude_ad, extra_terms=extra_terms)
        cfg = st.session_state["config"]
        cfg["geo_query_list"] = queries
        st.session_state["config"] = cfg
        add_log(f"[UI] GEO queries: {queries}")

        retmax = int(cfg.get("geo_retmax_each", 40))
        add_log(f"[UI] Search GEO retmax_each={retmax} ...")
        df = cached_geo_search(queries, retmax_each=retmax)
        st.session_state["candidates_df"] = df

    df = st.session_state.get("candidates_df", pd.DataFrame())
    if df is None or df.empty:
        st.info("No GEO candidates yet. Click **Build & Search GEO**.")
    else:
        st.write(f"Candidates: {df.shape[0]}")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download candidates CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="geo_candidates.csv", mime="text/csv")

# --- Tab 2: Validate ---
with tabs[1]:
    st.subheader("Download SOFT and validate case/control feasibility")
    df = st.session_state.get("candidates_df", pd.DataFrame())
    cfg = st.session_state["config"]

    if df is None or df.empty:
        st.info("Run GEO Search first.")
    else:
        # Choose a subset to validate
        gse_list = df["accession"].tolist()
        to_validate = st.multiselect("Select GSE to validate", options=gse_list, default=gse_list[: min(10, len(gse_list))])

        if st.button("Validate selected GSE"):
            st.session_state["logs"] = []
            rows = []
            min_reps = int(cfg.get("min_reps_per_group", 2))
            unk_strict = float(cfg.get("max_unknown_frac_strict", 0.1))
            unk_relaxed = float(cfg.get("max_unknown_frac_relaxed", 0.35))

            for i, gse in enumerate(to_validate):
                try:
                    add_log(f"[VALIDATE] {gse}: download+parse SOFT ...")
                    df_meta = cached_soft_parse(gse, project_dir=str(cfg.get("project_dir")))
                    df_l = label_conditions(df_meta, gse=gse, config=cfg)
                    counts = condition_counts(df_l)
                    strict_ok = pass_flags(counts, min_reps=min_reps, max_unknown_frac=unk_strict)
                    relaxed_ok = pass_flags(counts, min_reps=min_reps, max_unknown_frac=unk_relaxed)

                    rows.append({
                        "gse": gse,
                        "pass_strict": strict_ok,
                        "pass_relaxed": relaxed_ok,
                        "case": counts["case"],
                        "control": counts["control"],
                        "ambiguous": counts["ambiguous"],
                        "unknown": counts["unknown"],
                        "unknown_frac": (counts["unknown"] + counts["ambiguous"]) / max(1, counts["total"]),
                        "total": counts["total"],
                    })
                except Exception as e:
                    rows.append({"gse": gse, "error": repr(e)})

            df_val = pd.DataFrame(rows)
            st.session_state["validation_df"] = df_val

        df_val = st.session_state.get("validation_df", pd.DataFrame())
        if df_val is not None and not df_val.empty:
            st.dataframe(df_val, use_container_width=True)
            st.download_button("Download validation CSV", data=df_val.to_csv(index=False).encode("utf-8"), file_name="gse_validation.csv", mime="text/csv")

            # choose GSE to run
            gse_ok = df_val[df_val.get("pass_relaxed", False) == True]["gse"].tolist()
            sel = st.multiselect("Select GSE to run pipeline", options=df_val["gse"].tolist(), default=gse_ok[: min(5, len(gse_ok))])
            st.session_state["selected_gse"] = sel

# --- Tab 3: Run Pipeline ---
with tabs[2]:
    st.subheader("Run end-to-end pipeline for selected GSE")
    cfg = st.session_state["config"]
    sel = st.session_state.get("selected_gse", [])
    if not sel:
        st.info("Select GSE in **Validate & Select** tab first.")
    else:
        st.write("Selected GSE:", ", ".join(sel))

        if st.button("Run pipeline now"):
            st.session_state["logs"] = []
            st.session_state["run_results"] = []
            prog = st.progress(0)
            for i, gse in enumerate(sel):
                try:
                    add_log(f"\n=== RUN {gse} ({i+1}/{len(sel)}) ===")
                    res = run_one_gse(gse, cfg, logger=add_log)
                    st.session_state["run_results"].append(res)
                    add_log(f"[OK] {gse} done.")
                except Exception as e:
                    add_log(f"[ERROR] {gse}: {repr(e)}")
                    add_log(traceback.format_exc())
                    st.session_state["run_results"].append({"gse": gse, "error": repr(e)})
                prog.progress((i + 1) / len(sel))

            # Load per-GSE ranked files into a combined DF
            all_ranked = []
            for r in st.session_state["run_results"]:
                p = r.get("compound_ranked_csv")
                if p and Path(p).exists():
                    df_r = pd.read_csv(p)
                    all_ranked.append(df_r)
            if all_ranked:
                df_all = pd.concat(all_ranked, ignore_index=True)
                st.session_state["ranked_all_df"] = df_all
                st.session_state["aggregated_df"] = aggregate_across_gse(df_all)

# --- Tab 4: Results ---
with tabs[3]:
    st.subheader("Ranked compounds")

    df_all = st.session_state.get("ranked_all_df", pd.DataFrame())
    df_agg = st.session_state.get("aggregated_df", pd.DataFrame())
    runs = st.session_state.get("run_results", [])

    if runs:
        st.write("Run summary:")
        st.dataframe(pd.DataFrame(runs), use_container_width=True)

    if df_all is None or df_all.empty:
        st.info("No ranked results yet. Run pipeline first.")
    else:
        st.markdown("### Per-GSE ranked list (combined view)")
        st.dataframe(df_all, use_container_width=True)
        st.download_button("Download per-GSE ranked CSV", data=df_all.to_csv(index=False).encode("utf-8"), file_name="compound_ranked_all_gse.csv", mime="text/csv")

    if df_agg is not None and not df_agg.empty:
        st.markdown("### Aggregated across GSE")
        st.dataframe(df_agg, use_container_width=True)
        st.download_button("Download aggregated CSV", data=df_agg.to_csv(index=False).encode("utf-8"), file_name="compound_aggregated.csv", mime="text/csv")

# --- Logs ---
with tabs[4]:
    st.subheader("Logs")
    render_logs()
