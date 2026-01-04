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
        g = g.split(".")[0].split("//")[0].strip().upper()
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out

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
    
    # === 冲突优先 Control ===
    if hit_case and hit_ctrl:
        return "Control", "#e6ffe6" # Green
    
    if hit_case: return "Case", "#ffe6e6" # Red
    if hit_ctrl: return "Control", "#e6ffe6" # Green
    
    return "Unknown", "grey"

# --- 关键：智能矩阵读取器 ---
def smart_load_matrix(path):
    """
    智能查找表头行，解决 !Sample_title 被当做注释过滤的问题
    """
    header_row = None
    # 1. 扫描前1000行，寻找包含多个 GSM 的行
    with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i > 1000: break
            # 如果一行里包含 >1 个 "GSM"，很可能是表头
            if line.count("GSM") > 1:
                header_row = i
                break
    
    if header_row is None:
        # 兜底：假设它是 ID_REF 开头
        with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i > 1000: break
                if "ID_REF" in line or "id_ref" in line:
                    header_row = i
                    break
    
    # 2. 读取
    try:
        if header_row is not None:
            # 找到了表头，跳过前面的行
            df = pd.read_csv(path, sep="\t", skiprows=header_row, index_col=0, on_bad_lines='skip')
        else:
            # 没找到，尝试盲读（不要 comment='!'，否则会丢数据）
            df = pd.read_csv(path, sep="\t", comment=None, index_col=0, on_bad_lines='skip')
            
        # 3. 清洗
        # 删除所有列名不是 GSM 的列
        gsm_cols = [c for c in df.columns if "GSM" in str(c)]
        if not gsm_cols:
             # 有时候列名是 "Sample 1 (GSM123)"
             gsm_cols = [c for c in df.columns if re.search(r'GSM\d+', str(c))]
        
        df = df[gsm_cols]
        # 提取纯 GSM ID 作为列名
        new_cols = []
        for c in df.columns:
            m = re.search(r'(GSM\d+)', str(c))
            new_cols.append(m.group(1) if m else str(c))
        df.columns = new_cols
        
        # 强制转数字
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(how='all')
        
        return df
    except Exception as e:
        raise e

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
        return None, f"Download Error: {str(e)}"

    # 2. 分组解析
    df_meta, msg = extract_metadata_only(gse)
    if df_meta is None or df_meta.empty:
        return None, f"Metadata Parse Error: {msg}"
    
    conditions = {}
    for idx, row in df_meta.iterrows():
        group, _ = determine_group(row["Full_Description"], case_terms, ctrl_terms)
        if group == "Case": conditions[row["GSM"]] = "case"
        elif group == "Control": conditions[row["GSM"]] = "control"
    
    case_samps = [k for k,v in conditions.items() if v == "case"]
    ctrl_samps = [k for k,v in conditions.items() if v == "control"]
    
    if len(case_samps) == 0 or len(ctrl_samps) == 0:
        return None, f"Insufficient Samples: Case={len(case_samps)}, Ctrl={len(ctrl_samps)}"
    
    # 3. 智能读取矩阵
    try:
        df = smart_load_matrix(matrix_path)
        
        if not df.empty and df.max().max() > 50:
            df = np.log2(df + 1)
            
    except Exception as e:
        return None, f"Matrix Parse Error: {str(e)}"
    
    # 对齐
    valid_cols = [c for c in df.columns if c in conditions]
    
    if len(valid_cols) < 2:
        return None, f"Column Mismatch. Matrix GSMs: {list(df.columns[:3])}... vs Soft GSMs: {list(conditions.keys())[:3]}..."
    
    df = df[valid_cols]
    
    case_cols = [c for c in valid_cols if conditions[c] == "case"]
    ctrl_cols = [c for c in valid_cols if conditions[c] == "control"]
    
    if len(case_cols) < 1 or len(ctrl_cols) < 1:
         return None, f"Aligned Samples Missing: Case={len(case_cols)}, Ctrl={len(ctrl_cols)}"

    # 4. 差异分析
    results = []
    use_ttest = len(case_cols) >= 2 and len(ctrl_cols) >= 2
    
    for gene, row in df.iterrows():
        case_vals = row[case_cols].values
        ctrl_vals = row[ctrl_cols].values
        
        diff = np.mean(case_vals) - np.mean(ctrl_vals)
        p = 1.0
        
        if use_ttest:
            if np.std(case_vals) > 1e-9 and np.std(ctrl_vals) > 1e-9:
                try:
                    _, p = stats.ttest_ind(case_vals, ctrl_vals, equal_var=False)
                except: pass
        
        results.append({"gene": gene, "log2fc": diff, "pval": p})
        
    res_df = pd.DataFrame(results)
    if res_df.empty: return None, "No valid DE results"
    
    res_df["pval"] = res_df["pval"].fillna(1.0)
    res_df["padj"] = multipletests(res_df["pval"], method="fdr_bh")[1]
    res_df = res_df.sort_values("log2fc", key=abs, ascending=False)
    
    res_df["gene_symbol"] = res_df["gene"].apply(lambda x: str(x).split("//")[0].split(".")[0].strip().upper())
    
    return res_df, f"Success: Case={len(case_cols)}, Ctrl={len(ctrl_cols)}"

# --- API ---

def run_l1000fwd(up_genes, dn_genes):
    url = "https://maayanlab.cloud/l1000fwd/sig_search"
    payload = {"up_genes": up_genes[:150], "down_genes": dn_genes[:150]}
    try:
        r = requests.post(url, json=payload, timeout=30)
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
    st.info("💡 优先 Control: 如果样本同时包含 Case 和 Control 词汇，将强制判定为 Control。")
    
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
            
            # Colors
            def color_row(row):
                c = {"Case":'background-color:#ffe6e6', "Control":'background-color:#e6ffe6', "Unknown":'background-color:#f0f0f0'}
                return [c.get(row["Group"], "")] * len(row)

            st.dataframe(df_display.style.apply(color_row, axis=1), use_container_width=True)

with tab3:
    st.subheader("⚡ 批处理分析")
    if st.button("🚀 启动分析 (Start Pipeline)", type="primary"):
        results_bucket = []
        log_container = st.container()
        progress = st.progress(0)
        
        for i, gse in enumerate(st.session_state["selected_gses"]):
            with log_container:
                st.write(f"**Processing {gse}...**")
                df_de, msg = run_analysis_pipeline(gse, case_terms, ctrl_terms)
                
                if df_de is None:
                    st.error(f"❌ {gse} Failed: {msg}")
                    continue
                
                st.success(f"✅ {gse} DE Done. Genes: {len(df_de)}")
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
