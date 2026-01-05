\
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import pandas as pd


def _norm_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def get_gse_override(config: Dict[str, Any], gse: str) -> Dict[str, Any]:
    ov = (config.get("per_gse_overrides") or {}).get(gse.upper(), {}) or {}
    # normalize keys
    return {str(k).lower(): v for k, v in ov.items()}


def _match_any(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(term.lower() in t for term in terms if term)


def infer_condition_for_row(
    row: pd.Series,
    gse: str,
    config: Dict[str, Any],
) -> str:
    """
    Return: case | control | ambiguous | unknown
    Strategy:
      1) Use override 'use_field_first' if exists (description/characteristics/title/...)
      2) Use prefer_keys in characteristics_kv
      3) fallback to concatenated text across key columns
      4) fallback to title regex if configured
    """
    gse = gse.upper()
    ov = get_gse_override(config, gse)
    cond_cfg = config.get("condition", {}) or {}

    case_terms = list(ov.get("case_terms") or cond_cfg.get("case_terms") or [])
    control_terms = list(ov.get("control_terms") or cond_cfg.get("control_terms") or [])
    ambiguous_policy = str(cond_cfg.get("ambiguous_policy", "ambiguous")).lower()

    # 1) override field first
    field_first = ov.get("use_field_first")
    if field_first:
        txt = _norm_text(row.get(str(field_first)))
        is_case = _match_any(txt, case_terms)
        is_ctrl = _match_any(txt, control_terms)
        if is_case and not is_ctrl:
            return "case"
        if is_ctrl and not is_case:
            return "control"
        if is_case and is_ctrl:
            return ambiguous_policy if ambiguous_policy in ("case","control") else "ambiguous"

    # 2) prefer keys
    prefer_keys = [str(x).lower() for x in cond_cfg.get("prefer_keys", [])]
    kv = row.get("characteristics_kv") or {}
    if isinstance(kv, dict):
        for k in prefer_keys:
            if k in kv:
                txt = _norm_text(kv.get(k))
                is_case = _match_any(txt, case_terms)
                is_ctrl = _match_any(txt, control_terms)
                if is_case and not is_ctrl:
                    return "case"
                if is_ctrl and not is_case:
                    return "control"
                if is_case and is_ctrl:
                    return ambiguous_policy if ambiguous_policy in ("case","control") else "ambiguous"

    # 3) concat fallback
    txt = " | ".join([
        _norm_text(row.get("title")),
        _norm_text(row.get("source")),
        _norm_text(row.get("characteristics")),
        _norm_text(row.get("description")),
    ])
    is_case = _match_any(txt, case_terms)
    is_ctrl = _match_any(txt, control_terms)
    if is_case and not is_ctrl:
        return "case"
    if is_ctrl and not is_case:
        return "control"
    if is_case and is_ctrl:
        return ambiguous_policy if ambiguous_policy in ("case","control") else "ambiguous"

    # 4) title regex fallback
    title = str(row.get("title", "") or "")
    t_case_re = ov.get("title_case_regex")
    t_ctrl_re = ov.get("title_control_regex")
    try:
        if t_case_re and re.search(t_case_re, title):
            return "case"
        if t_ctrl_re and re.search(t_ctrl_re, title):
            return "control"
    except re.error:
        pass

    return "unknown"


def label_conditions(df_meta: pd.DataFrame, gse: str, config: Dict[str, Any]) -> pd.DataFrame:
    df = df_meta.copy()
    df["condition"] = df.apply(lambda r: infer_condition_for_row(r, gse=gse, config=config), axis=1)
    return df


def condition_counts(df_meta_labeled: pd.DataFrame) -> Dict[str, int]:
    vc = df_meta_labeled["condition"].value_counts(dropna=False).to_dict()
    out = {k: int(vc.get(k, 0)) for k in ["case", "control", "ambiguous", "unknown"]}
    out["total"] = int(len(df_meta_labeled))
    return out


def pass_flags(counts: Dict[str, int], min_reps: int, max_unknown_frac: float) -> bool:
    total = max(1, counts.get("total", 0))
    unk = counts.get("unknown", 0) + counts.get("ambiguous", 0)
    unk_frac = float(unk) / float(total)
    ok_groups = (counts.get("case", 0) >= min_reps) and (counts.get("control", 0) >= min_reps)
    ok_unk = unk_frac <= max_unknown_frac
    return bool(ok_groups and ok_unk)
