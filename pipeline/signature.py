\
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    import mygene  # type: ignore
except Exception:
    mygene = None


_ENSG_RE = re.compile(r"^ENSG\d+(\.\d+)?$", re.I)


def looks_like_ensembl(gene: str) -> bool:
    return bool(_ENSG_RE.match(str(gene).strip()))


def map_genes_to_symbol(genes: List[str], species: str = "human") -> Dict[str, str]:
    """
    Map Ensembl IDs to gene symbols using mygene.info.
    Returns dict {input -> symbol} for those resolved.
    """
    if mygene is None or not genes:
        return {}
    mg = mygene.MyGeneInfo()
    # querymany accepts mixed identifiers
    res = mg.querymany(genes, scopes=["ensembl.gene", "symbol"], fields="symbol", species=species, as_dataframe=False)
    out: Dict[str, str] = {}
    for r in res:
        q = str(r.get("query", ""))
        sym = r.get("symbol")
        if sym and isinstance(sym, str):
            out[q] = sym
    return out


def build_signature_from_de(
    df_de: pd.DataFrame,
    top_n: int = 150,
    min_abs_log2fc: float = 0.25,
    species: str = "human",
) -> Dict[str, List[str]]:
    """
    Build UP and DN gene lists from DE table.
    """
    df = df_de.copy()
    df = df.dropna(subset=["log2fc"])
    df = df[df["log2fc"].abs() >= float(min_abs_log2fc)]
    # If padj exists, prefer it
    if "padj" in df.columns:
        df = df.sort_values(["padj", "pval"], ascending=[True, True])
    elif "pval" in df.columns:
        df = df.sort_values("pval", ascending=True)

    up = df[df["log2fc"] > 0]["gene_raw"].astype(str).tolist()
    dn = df[df["log2fc"] < 0]["gene_raw"].astype(str).tolist()

    up = up[: top_n]
    dn = dn[: top_n]

    # Map Ensembl -> SYMBOL if needed
    ens = [g for g in up + dn if looks_like_ensembl(g)]
    if ens:
        mapping = map_genes_to_symbol(list(set(ens)), species=species)
        up = [mapping.get(g, g) for g in up]
        dn = [mapping.get(g, g) for g in dn]

    # Uppercase for connectivity APIs
    up = [g.upper() for g in up if g]
    dn = [g.upper() for g in dn if g]
    return {"up": up, "dn": dn}


def save_signature(sig: Dict[str, List[str]], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    up_p = out_dir / "UP.txt"
    dn_p = out_dir / "DN.txt"
    up_p.write_text("\n".join(sig.get("up", [])) + "\n")
    dn_p.write_text("\n".join(sig.get("dn", [])) + "\n")
    return up_p, dn_p
