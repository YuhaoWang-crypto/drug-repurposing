\
from __future__ import annotations

from typing import Any, Dict


def default_config() -> Dict[str, Any]:
    """
    Streamlit will show this as editable JSON.
    Keep keys aligned with your Colab pipeline as much as possible.
    """
    return {
        # Project dirs
        "project_dir": "./repurpose_pipeline_clcn",

        # GEO search
        "geo_retmax_each": 40,
        "geo_exclude_terms": ['Alzheimer', 'dementia', '"AD brain"'],

        # If you already know the GSEs, you can paste them here; otherwise leave empty
        "gse_list": [],

        # Query templates (auto-built in UI; you can override here)
        "geo_query_list": [],

        # Condition inference
        "condition": {
            "prefer_keys": [
                "diagnosis", "genotype", "disease state", "disease", "condition",
                "group", "construct", "treatment"
            ],
            "case_terms": [
                "mutation", "mutant", "variant", "patient",
                "ko", "knockout", "knock-down", "knockdown",
                "crispr", "sirna", "shrna", "overexpression", "oe",
                # CLCN-specific hints
                "clcn2 mutation", "clcn3 knockout", "clcn7", "clcn5",
            ],
            "control_terms": [
                "control", "wt", "wild type", "wild-type", "healthy",
                "normal", "untreated", "vehicle",
            ],
            # ambiguous | case | control
            "ambiguous_policy": "ambiguous",
        },

        # Per-GSE overrides (optional). Regex strings are used for title/column fallbacks.
        "per_gse_overrides": {
            "GSE216834": {
                "use_field_first": "description",
                "case_terms": ["clcn2 mutation", "patient", "mutant"],
                "control_terms": ["control", "wt", "wild type"],
                "title_case_regex": r"^M\d+hiPSC",
                "title_control_regex": r"^C\d+hiPSC",
                # If bulk columns are NOT GSMs, allow mapping by these regex
                "col_case_regex": r"^M\d+_",
                "col_control_regex": r"^C\d+_",
            },
        },

        # Validation thresholds
        "min_reps_per_group": 2,
        "max_unknown_frac_strict": 0.1,
        "max_unknown_frac_relaxed": 0.35,

        # Bulk DE parameters
        "de_min_samples_total": 4,
        "de_eps": 1e-6,

        # Signature parameters
        "signature_top_n": 150,
        "sig_min_abs_log2fc": 0.25,

        # Connectivity toggles
        "run_l1000cds2": True,
        "run_l1000fwd": True,
        "run_enrichr": True,

        # L1000CDS2
        "l1000_mode": "reverse",
        "l1000_urls": [
            "https://maayanlab.cloud/L1000CDS2/query",
            "https://amp.pharm.mssm.edu/L1000CDS2/query",
        ],
        "l1000_timeout": 120,
        "l1000_limit": 50,

        # L1000FWD
        "l1000fwd_base_urls": [
            "https://maayanlab.cloud/l1000fwd",
            "https://maayanlab.cloud/L1000FWD",
        ],
        "l1000fwd_timeout": 120,
        "l1000fwd_limit": 50,

        # Enrichr
        "enrichr_base_url": "https://maayanlab.cloud/Enrichr",
        "enrichr_libraries": [
            "LINCS_L1000_Chem_Pert_up",
            "LINCS_L1000_Chem_Pert_down",
            "Drug_Perturbations_from_GEO_up",
            "Drug_Perturbations_from_GEO_down",
        ],

        # Ranking weights
        "rank_weights": {"cds2": 1.0, "l1000fwd": 1.0, "enrichr": 0.7},

        # Module definition for optional PPI check:
        # - Option A (recommended): manually provide module gene symbols (gene names)
        "module_gene_symbols": [
            # Example starting set (edit/delete as needed)
            "CLCN2", "CLCN3", "CLCN5", "CLCN7",
            "LAMP1", "LAMP2",
            "CTSB", "CTSD",
            "TFEB", "TFE3",
            "MTOR", "RAB7A",
            "ATP6V0A1", "ATP6V1B2",
        ],
        # - Option B: GO terms (currently NOT auto-mapped to gene symbols in this template)
        "module_go_terms": ["GO:0005764", "GO:0006914", "GO:0006811"],  # lysosome, autophagy, ion transport
        "quickgo_taxon": "9606",

        # Optional PPI edges CSV (uploaded in UI)
        "ppi_edges_csv": "",
        "ppi_max_dist": 2,
    }
