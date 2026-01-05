\
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd
import requests

from .utils import requests_get


def load_ppi_edges(edges_csv: Path) -> nx.Graph:
    """
    edges_csv: two columns a,b (gene symbols).
    """
    G = nx.Graph()
    df = pd.read_csv(edges_csv)
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError("PPI edges CSV must have at least two columns.")
    a, b = cols[0], cols[1]
    for x, y in zip(df[a].astype(str), df[b].astype(str)):
        x = x.strip().upper()
        y = y.strip().upper()
        if not x or not y or x == "NAN" or y == "NAN":
            continue
        if x == y:
            continue
        G.add_edge(x, y)
    return G


def quickgo_fetch_gene_products(go_id: str, taxon: str = "9606", limit: int = 200, max_pages: int = 50, timeout: int = 60) -> List[str]:
    """
    QuickGO annotation search (best-effort).
    Returns list of geneProductId strings; convert to symbols is non-trivial, so we keep IDs.
    Many users will prefer uploading module genes instead.
    """
    base = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
    genes: List[str] = []
    page = 1
    for _ in range(max_pages):
        params = {
            "goId": go_id,
            "taxonId": taxon,
            "limit": int(limit),
            "page": int(page),
        }
        r = requests.get(base, params=params, headers={"Accept": "application/json"}, timeout=timeout)
        if r.status_code != 200:
            break
        js = r.json()
        results = js.get("results") or []
        if not isinstance(results, list) or not results:
            break
        for it in results:
            gp = it.get("geneProductId") or it.get("geneProduct", {}).get("id")
            if gp:
                genes.append(str(gp))
        # pagination
        pi = js.get("pageInfo") or {}
        if pi.get("isLast") is True:
            break
        page += 1
    # dedupe
    genes_u = sorted(set(genes))
    return genes_u


def ppi_module_metrics(
    G: nx.Graph,
    module_genes: Set[str],
    sig_up: Set[str],
    sig_dn: Set[str],
    max_dist: int = 2,
) -> Dict[str, float]:
    """
    Compute simple overlap and distance-to-module stats for signature genes.
    """
    module = {g.upper() for g in module_genes if g}
    up = {g.upper() for g in sig_up if g}
    dn = {g.upper() for g in sig_dn if g}
    sig = up | dn

    # overlap
    ov_up = len(module & up)
    ov_dn = len(module & dn)
    ov_sig = len(module & sig)

    # distance: BFS from module
    dist = {}
    # Keep only module nodes present in graph
    sources = [g for g in module if g in G]
    if sources:
        dist = nx.multi_source_shortest_path_length(G, sources, cutoff=max_dist)
    near_up = sum(1 for g in up if g in dist)
    near_dn = sum(1 for g in dn if g in dist)
    near_sig = sum(1 for g in sig if g in dist)

    return {
        "module_size": float(len(module)),
        "sig_up_n": float(len(up)),
        "sig_dn_n": float(len(dn)),
        "overlap_up": float(ov_up),
        "overlap_dn": float(ov_dn),
        "overlap_sig": float(ov_sig),
        "near_up_leq_dist": float(near_up),
        "near_dn_leq_dist": float(near_dn),
        "near_sig_leq_dist": float(near_sig),
    }
