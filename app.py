import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import os
import time
import gzip
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import mygene
from urllib.parse import quote
from collections import Counter

# ==========================================
# 0. 配置与初始化
# ==========================================
st.set_page_config(page_title="GEO Drug Repurposing Pipeline", layout="wide", page_icon="💊")

WORK_DIR = Path("workspace")
RAW_DIR = WORK_DIR / "raw"
PROC_DIR = WORK_DIR / "proc"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

if "geo_hits" not in st.session_state:
    st.session_state["geo_hits"] = pd.DataFrame()
if "selected_gses" not in st.session_state:
    st.session_state["selected_gses"] = []
if "metadata_cache" not in st.session_state:
    st.session_state["metadata_cache"] = {}

# ==========================================
# 1. 核心工具函数
# ==========================================

@st.cache_resource
def get_mygene_info():
    return mygene.MyGeneInfo()

def clean_gene_list(genes):
    out = []
    seen = set()
    for g in genes:
        if not isinstance(g, str): continue
        # 去除 Ensembl 版本号 (ENSG000.1 -> ENSG000) 和 ///
        g = g.split(".")[0].split("//")[0].strip().upper()
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out

# --- Map Ensembl to Symbol (简单的批量映射，解决 ENSG 问题) ---
@st.cache_data
def map_ensembl_to_symbol_simple(gene_list):
    # 如果大部分已经是 Symbol (不含 ENSG)，直接返回
    if not gene_list: return {}
    sample = gene_list[:10]
    if not any(x.startswith("ENS") for x in sample):
        return {g: g for g in gene_list}
        
    mg = get_mygene_info()
    res = mg.querymany(gene_list, scopes='ensembl.gene', fields='symbol', species='human', returnall=False, verbose=False)
    mapping = {}
    for r in res:
        query = r.get('query')
        symbol = r.get('symbol')
        if query and symbol:
            mapping[query] = symbol.upper()
    return mapping

# --- GEO 下载与解析 ---

def geo_search(query, retmax=30):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        search_url = f"{base}/esearch.fcgi?db=gds&term={quote(query)}&retmax={retmax}&retmode=json"
        r = requests.get(search_url, timeout=10).json()
        ids = r.get("esearchresult", {}).get("idlist", [])
        if not ids: return pd.DataFrame()
        
        summary_url = f"{base}/esummary.fcgi?db=gds&id={','.join(ids)}&retmode=json"
        r = requests.get(summary_url, timeout=10).json()
        result = r.get("result", {})
        
        rows = []
        for uid in ids:
            if uid not in result: continue
            item = result[uid]
            acc = item.get("accession", "")
            if not acc.startswith("GSE"): continue
            rows.append({
                "Accession": acc,
                "Title": item.get("title", ""),
                "Summary": item.get("summary", "")[:200] + "...",
                "Taxon": item.get("taxon", ""),
                "Samples": item.get("n_samples", 0),
                "Date": item.get("pdat", "")
            })
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame()

def download_file(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0: return
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)
    except Exception as e:
        if path.exists(): path.unlink()
        raise e

def get_geo_urls(gse):
    gse = gse.strip().upper()
    num = re.findall(r'\d+', gse)
    if not num: return "", ""
    series_id = int(num[0])
    prefix = f"GSE{series_id // 1000}nnn"
    soft_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/soft/{gse}_family.soft.gz"
    matrix_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/matrix/{gse}_series_matrix.txt.gz"
    return soft_url, matrix_url

def extract_metadata_only(gse):
    gse_dir = RAW_DIR / gse
    gse_dir.mkdir(exist_ok=True)
    soft_url, _ = get_geo_urls(gse)
    soft_path = gse_dir / f"{gse}_family.soft.gz"
    try:
        download_file(soft_url, soft_path)
    except Exception as e:
        return None, str(e)

    meta = []
    current_gsm = None
    current_data = {"GSM": "", "Title": "", "Text": []}
    
    with gzip.open(soft_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith("^SAMPLE ="):
                if current_gsm:
                    meta.append({
                        "GSM": current_data["GSM"],
                        "Title": current_data["Title"],
                        "Full_Description": " | ".join(current_data["Text"]).lower()
                    })
                current_gsm = line.split("=")[1].strip()
                current_data = {"GSM": current_gsm, "Title": "", "Text": []}
            elif current_gsm:
                if line.startswith("!Sample_title"):
                    current_data["Title"] = line.split("=", 1)[1].strip()
                    current_data["Text"].append(line.split("=", 1)[1].strip())
                elif line.startswith(("!Sample_source_name", "!Sample_characteristics", "!Sample_description")):
                    try:
                        content = line.split("=", 1)[1].strip()
                        current_data["Text"].append(content)
                    except: pass
        if current_gsm:
             meta.append({
                "GSM": current_data["GSM"],
                "Title": current_data["Title"],
                "Full_Description": " | ".join(current_data["Text"]).lower()
            })
    return pd.DataFrame(meta), "Success"

def determine_group(text, case_terms, ctrl_terms):
    text = text.lower()
    hit_case = any(t in text for t in case_terms)
    hit_ctrl = any(t in text for t in ctrl_terms)
    # 冲突优先判定为 Control
    if hit_case and hit_ctrl: return "Control", "#e6ffe6"
    if hit_case: return "Case", "#ffe6e6"
    if hit_ctrl: return "Control", "#e6ffe6"
    return "Unknown", "grey"

# --- 关键修复：灵活的矩阵读取 & 样本对齐 ---

def normalize_str(s):
    """去除标点符号、空格，转小写，用于模糊匹配"""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def smart_load_matrix(path):
    """
    不强制过滤 GSM，保留所有列，把对齐工作留给后面
    """
    header_row = None
    with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i > 2000: break
            if "!series_matrix_table_begin" in line:
                header_row = i + 1
                break
            if line.startswith("\"ID_REF\"") or line.startswith("ID_REF"):
                header_row = i
                break
    
    try:
        skip = header_row if header_row is not None else "infer"
        if skip == "infer":
            df = pd.read_csv(path, sep="\t", comment="!", index_col=0, on_bad_lines='skip')
        else:
            df = pd.read_csv(path, sep="\t", skiprows=skip, index_col=0, on_bad_lines='skip')
            
        # 简单清洗列名：去掉引号
        df.columns = [str(c).strip().replace('"', '').replace("'", "") for c in df.columns]
        
        # 强制转数字
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(how='all')
        return df
    except Exception as e:
        raise e

def align_samples(df_matrix, df_meta, conditions):
    """
    尝试将 Matrix 的列名 对齐到 Soft 的 GSM
    策略 1: 直接 GSM 匹配
    策略 2: 用 Sample Title 模糊匹配
    """
    matrix_cols = df_matrix.columns.tolist()
    
    # 策略 1: 直接匹配 GSM
    matched_gsm = [c for c in matrix_cols if c in conditions]
    if len(matched_gsm) >= 2:
        return df_matrix[matched_gsm], {c: conditions[c] for c in matched_gsm}, "Direct GSM Match"
    
    # 策略 2: Title 匹配 (针对 GSE216834 这种列名是 Title 的情况)
    # 构建 Title -> GSM -> Condition 的映射
    title_map = {} # normalized_title -> gsm
    for idx, row in df_meta.iterrows():
        norm_title = normalize_str(row["Title"])
        if norm_title:
            title_map[norm_title] = row["GSM"]
            
    final_cols = []
    final_conditions = {}
    
    for col in matrix_cols:
        norm_col = normalize_str(col)
        # 尝试完全包含匹配 (比如 matrix列名是 "C1hiPSC_RPE_D28_rep1", title 是 "C1hiPSC_RPE, D28, rep1")
        # 这里的 norm_col 会变成 c1hipscrped28rep1
        
        # 查找 title_map 中是否有 key 等于 norm_col
        if norm_col in title_map:
            gsm = title_map[norm_col]
            if gsm in conditions:
                final_cols.append(col)
                final_conditions[col] = conditions[gsm]
    
    if len(final_cols) >= 2:
        return df_matrix[final_cols], final_conditions, "Title Fuzzy Match"

    return None, None, "Alignment Failed"

# --- 差异分析主流程 ---

def run_analysis_pipeline(gse, case_terms, ctrl_terms):
    gse_dir = RAW_DIR / gse
    gse_dir.mkdir(exist_ok=True)
    
    soft_url, matrix_url = get_geo_urls(gse)
    soft_path = gse_dir / f"{gse}_family.soft.gz"
    matrix_path = gse_dir / f"{gse}_series_matrix.txt.gz"
    
    # 1. 下载
    try:
        download_file(soft_url, soft_path)
        download_file(matrix_url, matrix_path)
    except Exception as e:
        return None, f"Download Error: {str(e)}", None

    # 2. 分组解析 (Soft)
    df_meta, msg = extract_metadata_only(gse)
    if df_meta is None or df_meta.empty:
        return None, f"Metadata Parse Error: {msg}", None
    
    conditions = {}
    for idx, row in df_meta.iterrows():
        clean_gsm = str(row["GSM"]).strip()
        group, _ = determine_group(row["Full_Description"], case_terms, ctrl_terms)
        if group == "Case": conditions[clean_gsm] = "case"
        elif group == "Control": conditions[clean_gsm] = "control"
    
    # 3. 读取矩阵
    try:
        df = smart_load_matrix(matrix_path)
        if not df.empty and df.max().max() > 50:
            df = np.log2(df + 1)
    except Exception as e:
        return None, f"Matrix Parse Error: {str(e)}", None
    
    # 4. 对齐 (核心修复点)
    df_aligned, col_conditions, mode = align_samples(df, df_meta, conditions)
    
    if df_aligned is None:
        return None, f"Column Mismatch. Matrix cols: {list(df.columns[:3])}... vs Soft GSMs. Tried Direct & Title match.", None
    
    case_cols = [c for c, cond in col_conditions.items() if cond == "case"]
    ctrl_cols = [c for c, cond in col_conditions.items() if cond == "control"]
    
    if len(case_cols) < 1 or len(ctrl_cols) < 1:
         return None, f"Aligned Samples Missing: Case={len(case_cols)}, Ctrl={len(ctrl_cols)}", None

    # 5. 差异分析
    use_ttest = len(case_cols) >= 2 and len(ctrl_cols) >= 2
    
    case_vals = df_aligned[case_cols].values
    ctrl_vals = df_aligned[ctrl_cols].values
    
    log2fc = np.nanmean(case_vals, axis=1) - np.nanmean(ctrl_vals, axis=1)
    
    pvals = np.ones(len(df_aligned))
    if use_ttest:
        with np.errstate(divide='ignore', invalid='ignore'):
            _, pvals = stats.ttest_ind(case_vals, ctrl_vals, axis=1, equal_var=False)
    
    res_df = pd.DataFrame({
        "gene": df_aligned.index,
        "log2fc": log2fc,
        "pval": np.nan_to_num(pvals, nan=1.0)
    })
    
    res_df = res_df.dropna(subset=["log2fc"])
    if res_df.empty: return None, "No valid DE results", None
    
    res_df["padj"] = multipletests(res_df["pval"], method="fdr_bh")[1]
    res_df = res_df.sort_values("log2fc", key=abs, ascending=False)
    
    # 6. ID 转换 (ENSG -> Symbol)
    res_df["gene_clean"] = res_df["gene"].apply(lambda x: str(x).split("//")[0].split(".")[0].strip().upper())
    
    # 检查是否全是 Ensembl ID，如果是，尝试映射
    top_genes = res_df["gene_clean"].head(10).tolist()
    if any(g.startswith("ENS") for g in top_genes):
        # 只有在需要的时候才调用 Mapping，节省时间
        all_genes = res_df["gene_clean"].tolist()
        mapping = map_ensembl_to_symbol_simple(all_genes)
        res_df["gene_symbol"] = res_df["gene_clean"].map(mapping).fillna(res_df["gene_clean"])
    else:
        res_df["gene_symbol"] = res_df["gene_clean"]
        
    # 移除没有 Symbol 的行
    res_df = res_df[res_df["gene_symbol"] != ""]
    
    return res_df, f"Success ({mode}): Case={len(case_cols)}, Ctrl={len(ctrl_cols)}", res_df

# --- API ---

def run_l1000fwd(up_genes, dn_genes):
    url = "https://maayanlab.cloud/l1000fwd/sig_search"
    payload = {"up_genes": up_genes[:150], "down_genes": dn_genes[:150]}
    try:
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code != 200: return pd.DataFrame()
        res_id = r.json().get("result_id")
        if not res_id: return pd.DataFrame()
        time.sleep(1)
        r2 = requests.get(f"https://maayanlab.cloud/l1000fwd/result/topn/{res_id}", timeout=30)
        data = r2.json()
        rows = []
        if "opposite" in data:
            for item in data["opposite"]:
                rows.append({"drug": item.get("pert_id"), "score": item.get("score"), "source": "L1000FWD"})
        return pd.DataFrame(rows)
    except: return pd.DataFrame()

def run_enrichr(genes, library="LINCS_L1000_Chem_Pert_down"):
    base = "https://maayanlab.cloud/Enrichr"
    try:
        r = requests.post(f"{base}/addList", files={'list': (None, '\n'.join(genes[:300])), 'description': (None, 'Streamlit')}, timeout=30)
        uid = r.json().get("userListId")
        if not uid: return pd.DataFrame()
        r2 = requests.get(f"{base}/enrich?userListId={uid}&backgroundType={library}", timeout=30)
        data = r2.json()
        if library not in data: return pd.DataFrame()
        rows = []
        for item in data[library]:
            rows.append({"drug": item[1].split("_")[0].split(" ")[0], "score": item[4], "source": "Enrichr"})
        return pd.DataFrame(rows)
    except: return pd.DataFrame()

# ==========================================
# 2. UI
# ==========================================

with st.sidebar:
    st.header("⚙️ 全局设置")
    
    default_case = "mutation, mutant, variant, patient, knockout, knockdown, disease, clcn, cf, cystic fibrosis, tumor, cancer, treated, stimulation, infected"
    default_ctrl = "control, wt, wild type, wild-type, healthy, normal, vehicle, pbs, dmso, mock, baseline, untreated, placebo, non-targeting"
    
    case_input = st.text_area("Case 关键词", default_case, height=120)
    ctrl_input = st.text_area("Control 关键词", default_ctrl, height=120)
    
    case_terms = [x.strip().lower() for x in case_input.split(",") if x.strip()]
    ctrl_terms = [x.strip().lower() for x in ctrl_input.split(",") if x.strip()]
    
    st.divider()
    top_n_genes = st.number_input("Top Genes", 50, 500, 150)
    taxon_filter = st.selectbox("Species", ["Homo sapiens", "Mus musculus", "All"], index=0)

tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ 搜索数据集", "2️⃣ 🔬 样本分组调试器", "3️⃣ ⚡ 运行分析", "4️⃣ 📊 结果看板"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        query_text = st.text_input("GEO Query", value='(CLCN2 OR "chloride channel 2") AND (mutation OR knockout) AND "RNA-seq"')
    with col2:
        if st.button("Search"):
            with st.spinner("Searching..."):
                df = geo_search(query_text)
                if not df.empty and taxon_filter != "All": df = df[df["Taxon"] == taxon_filter]
                st.session_state["geo_hits"] = df

    if not st.session_state["geo_hits"].empty:
        hits = st.session_state["geo_hits"].copy()
        hits.insert(0, "Select", False)
        edited = st.data_editor(hits, column_config={"Select": st.column_config.CheckboxColumn(required=True)}, disabled=["Accession", "Title"], use_container_width=True, hide_index=True)
        st.session_state["selected_gses"] = edited[edited["Select"]]["Accession"].tolist()
        if st.session_state["selected_gses"]: st.success(f"Selected: {st.session_state['selected_gses']}")

with tab2:
    st.subheader("🔬 样本元数据调试")
    if not st.session_state["selected_gses"]:
        st.warning("Please select datasets in Tab 1.")
    else:
        inspect_gse = st.selectbox("Select Dataset:", st.session_state["selected_gses"])
        if st.button(f"Load Metadata for {inspect_gse}"):
            with st.spinner("Loading..."):
                df_meta, msg = extract_metadata_only(inspect_gse)
                if df_meta is not None: st.session_state["metadata_cache"][inspect_gse] = df_meta
        
        if inspect_gse in st.session_state["metadata_cache"]:
            df_display = st.session_state["metadata_cache"][inspect_gse].copy()
            df_display["Group"] = df_display["Full_Description"].apply(lambda x: determine_group(x, case_terms, ctrl_terms)[0])
            
            def color_row(row):
                c = {"Case":'background-color:#ffe6e6', "Control":'background-color:#e6ffe6', "Unknown":'background-color:#f0f0f0'}
                return [c.get(row["Group"], "")] * len(row)

            st.dataframe(df_display.style.apply(color_row, axis=1), use_container_width=True)
            
            st.divider()
            with st.expander("💡 关键词推荐"):
                all_text = " ".join(df_display["Full_Description"].astype(str).tolist()).lower()
                words = re.findall(r'\b[a-z]{3,}\b', all_text)
                stops = set(["the","and","for","with","sample","total","rna","homo","sapiens","description","characteristics","source","protocol","extraction","library","sequencing"])
                clean_words = [w for w in words if w not in stops and w not in case_terms and w not in ctrl_terms]
                common = Counter(clean_words).most_common(20)
                st.code(", ".join([f"{w[0]}" for w in common]), language="text")

with tab3:
    st.subheader("⚡ 批处理分析")
    if st.button("🚀 启动分析 (Start Pipeline)", type="primary"):
        results_bucket = []
        log_container = st.container()
        progress = st.progress(0)
        
        for i, gse in enumerate(st.session_state["selected_gses"]):
            with log_container:
                st.write(f"**Processing {gse}...**")
                # 接收第3个返回值 res_df
                df_de, msg, full_res_df = run_analysis_pipeline(gse, case_terms, ctrl_terms)
                
                if df_de is None:
                    st.error(f"❌ {gse} Failed: {msg}")
                    continue
                
                st.success(f"✅ {gse} DE Done. Genes: {len(df_de)} ({msg})")
                
                # === 用户需求：展示详细分析表 ===
                with st.expander(f"📊 点击查看 {gse} 差异分析详细结果"):
                    st.dataframe(full_res_df.head(100), use_container_width=True)
                
                up = df_de[df_de["log2fc"] > 0].head(top_n_genes)["gene_symbol"].tolist()
                dn = df_de[df_de["log2fc"] < 0].tail(top_n_genes)["gene_symbol"].tolist()
                
                if len(up)<5: continue
                
                df_l = run_l1000fwd(clean_gene_list(up), clean_gene_list(dn))
                df_e = run_enrichr(clean_gene_list(up), "LINCS_L1000_Chem_Pert_down")
                comb = pd.concat([df_l, df_e])
                if not comb.empty:
                    comb["gse"] = gse
                    results_bucket.append(comb)
            progress.progress((i+1)/len(st.session_state["selected_gses"]))
        
        if results_bucket:
            st.session_state["final_drug_rank"] = pd.concat(results_bucket)
            st.success("Done!")

with tab4:
    res = st.session_state.get("final_drug_rank", pd.DataFrame())
    if not res.empty:
        res["drug_clean"] = res["drug"].astype(str).str.lower().str.strip()
        agg = res.groupby("drug_clean").agg(
            Count=('gse', 'nunique'),
            Score_Sum=('score', 'sum'),
            GSEs=('gse', lambda x: ",".join(set(x)))
        ).reset_index().sort_values(["Count", "Score_Sum"], ascending=[False, False])
        st.dataframe(agg, use_container_width=True)
        st.download_button("Download CSV", agg.to_csv().encode("utf-8"), "drugs.csv")
