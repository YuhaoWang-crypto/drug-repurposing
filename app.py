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
st.set_page_config(page_title="GEO Pipeline (Diagnostic Mode)", layout="wide", page_icon="🩺")

WORK_DIR = Path("workspace")
RAW_DIR = WORK_DIR / "raw"
PROC_DIR = WORK_DIR / "proc"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

if "geo_hits" not in st.session_state: st.session_state["geo_hits"] = pd.DataFrame()
if "selected_gses" not in st.session_state: st.session_state["selected_gses"] = []
if "metadata_cache" not in st.session_state: st.session_state["metadata_cache"] = {}

# ==========================================
# 1. 核心工具函数
# ==========================================

@st.cache_resource
def get_mygene_info(): return mygene.MyGeneInfo()

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

# --- GEO 下载 ---
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
            rows.append({"Accession": acc, "Title": item.get("title", ""), "Summary": item.get("summary", "")[:200]+"...", "Taxon": item.get("taxon", ""), "Samples": item.get("n_samples", 0), "Date": item.get("pdat", "")})
        return pd.DataFrame(rows)
    except: return pd.DataFrame()

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
    try: download_file(soft_url, soft_path)
    except Exception as e: return None, str(e)

    meta = []
    current_gsm = None
    current_data = {"GSM": "", "Title": "", "Text": []}
    with gzip.open(soft_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith("^SAMPLE ="):
                if current_gsm:
                    meta.append({"GSM": current_data["GSM"], "Title": current_data["Title"], "Full_Description": " | ".join(current_data["Text"]).lower()})
                current_gsm = line.split("=")[1].strip()
                current_data = {"GSM": current_gsm, "Title": "", "Text": []}
            elif current_gsm:
                if line.startswith("!Sample_title"):
                    current_data["Title"] = line.split("=", 1)[1].strip()
                    current_data["Text"].append(line.split("=", 1)[1].strip())
                elif line.startswith(("!Sample_source_name", "!Sample_characteristics", "!Sample_description")):
                    try: current_data["Text"].append(line.split("=", 1)[1].strip())
                    except: pass
        if current_gsm:
             meta.append({"GSM": current_data["GSM"], "Title": current_data["Title"], "Full_Description": " | ".join(current_data["Text"]).lower()})
    return pd.DataFrame(meta), "Success"

def determine_group(text, case_terms, ctrl_terms):
    text = text.lower()
    hit_case = any(t in text for t in case_terms)
    hit_ctrl = any(t in text for t in ctrl_terms)
    if hit_case and hit_ctrl: return "Control", "#e6ffe6"
    if hit_case: return "Case", "#ffe6e6"
    if hit_ctrl: return "Control", "#e6ffe6"
    return "Unknown", "grey"

# --- 深度诊断版：矩阵读取与分析 ---

def run_analysis_pipeline_diagnostic(gse, case_terms, ctrl_terms):
    """
    返回: (df_de, error_message, full_result_df, LOGS_LIST)
    """
    logs = []
    def log(msg):
        timestamp = time.strftime("%H:%M:%S")
        logs.append(f"[{timestamp}] {msg}")

    log(f"开始处理 {gse}...")
    
    gse_dir = RAW_DIR / gse
    gse_dir.mkdir(exist_ok=True)
    soft_url, matrix_url = get_geo_urls(gse)
    soft_path = gse_dir / f"{gse}_family.soft.gz"
    matrix_path = gse_dir / f"{gse}_series_matrix.txt.gz"
    
    # 1. 下载
    log("Step 1: 检查文件...")
    try:
        download_file(soft_url, soft_path)
        download_file(matrix_url, matrix_path)
        log(f"文件就绪。Soft大小: {soft_path.stat().st_size/1024:.1f}KB, Matrix大小: {matrix_path.stat().st_size/1024:.1f}KB")
    except Exception as e:
        return None, f"下载失败: {e}", None, logs

    # 2. 解析元数据
    log("Step 2: 解析样本分组...")
    df_meta, msg = extract_metadata_only(gse)
    if df_meta is None or df_meta.empty:
        log(f"Soft解析失败: {msg}")
        return None, "Soft解析失败", None, logs
    
    conditions = {}
    for idx, row in df_meta.iterrows():
        clean_gsm = str(row["GSM"]).strip()
        group, _ = determine_group(row["Full_Description"], case_terms, ctrl_terms)
        if group in ["Case", "Control"]:
            conditions[clean_gsm] = group.lower()
    
    log(f"分组结果: 找到 {len(conditions)} 个有效样本 (Case/Control)。")
    if not conditions:
        log("❌ 严重错误: 没有样本被分到 Case 或 Control 组。请检查关键词。")
        return None, "分组失败: 0样本匹配", None, logs

    # 3. 读取矩阵 (最容易出错的地方)
    log("Step 3: 读取表达矩阵...")
    try:
        # 寻找表头
        header_row = None
        with gzip.open(matrix_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i > 2000: break
                if "!series_matrix_table_begin" in line:
                    header_row = i + 1
                    log(f"  -> 在第 {i+1} 行找到 table_begin 标记")
                    break
                if line.startswith("\"ID_REF\"") or line.startswith("ID_REF"):
                    header_row = i
                    log(f"  -> 在第 {i} 行找到 ID_REF")
                    break
        
        # 读取
        skip = header_row if header_row is not None else "infer"
        if skip == "infer":
            df = pd.read_csv(matrix_path, sep="\t", comment="!", index_col=0, on_bad_lines='skip')
        else:
            df = pd.read_csv(matrix_path, sep="\t", skiprows=skip, index_col=0, on_bad_lines='skip')
        
        log(f"  -> 原始矩阵形状: {df.shape}")
        log(f"  -> 原始前5列名: {list(df.columns[:5])}")

        # 清洗列名
        clean_cols_map = {}
        for c in df.columns:
            # 提取 GSM
            m = re.search(r'(GSM\d+)', str(c))
            if m: clean_cols_map[c] = m.group(1)
        
        if not clean_cols_map:
            # 尝试另一种策略：也许列名就是 GSM，只是带了引号
            for c in df.columns:
                clean = str(c).strip().replace('"', '').replace("'", "")
                if clean.startswith("GSM"):
                    clean_cols_map[c] = clean
        
        if not clean_cols_map:
            log("❌ 矩阵列名解析失败: 无法从列名中提取 GSM ID。")
            return None, "列名无GSM ID", None, logs
            
        df = df.rename(columns=clean_cols_map)
        # 去重
        df = df.loc[:, ~df.columns.duplicated()]
        log(f"  -> 清洗后包含GSM的列数: {len(df.columns)}")
        
        # 4. 对齐
        log("Step 4: 样本对齐...")
        common = set(df.columns).intersection(set(conditions.keys()))
        log(f"  -> Matrix中的GSM: {list(df.columns)[:3]}...")
        log(f"  -> Metadata中的GSM: {list(conditions.keys())[:3]}...")
        log(f"  -> 交集样本数: {len(common)}")
        
        if len(common) < 2:
            log("❌ 对齐失败: Matrix 和 Metadata 没有足够的共同样本。")
            return None, "样本对齐失败", None, logs
            
        df = df[list(common)]
        
        # 5. 数据清洗与转换
        log("Step 5: 数据数值化检查...")
        # 强制转数字
        df = df.apply(pd.to_numeric, errors='coerce')
        # 删除全是 NaN 的行
        orig_genes = len(df)
        df = df.dropna(axis=0, how='all')
        log(f"  -> 删除了 {orig_genes - len(df)} 行全空基因")
        
        if df.empty:
            log("❌ 错误: 矩阵在转数值后为空。")
            return None, "矩阵为空", None, logs
            
        # Log2 处理
        max_val = df.max().max()
        log(f"  -> 矩阵最大值: {max_val:.2f}")
        if max_val > 50:
            log("  -> 检测到非Log数据，执行 Log2(x+1) 变换...")
            df = np.log2(df + 1)
        
        # 6. 差异分析
        log("Step 6: 计算差异表达...")
        case_cols = [c for c in df.columns if conditions[c] == 'case']
        ctrl_cols = [c for c in df.columns if conditions[c] == 'control']
        log(f"  -> 最终分析样本: Case={len(case_cols)}, Control={len(ctrl_cols)}")
        
        if len(case_cols) < 2 or len(ctrl_cols) < 2:
            log("❌ 错误: 有效样本不足 (每组需>=2)。")
            return None, "样本不足", None, logs
            
        # Numpy 计算
        case_vals = df[case_cols].values
        ctrl_vals = df[ctrl_cols].values
        
        # 方差检查
        case_var = np.var(case_vals, axis=1)
        ctrl_var = np.var(ctrl_vals, axis=1)
        valid_genes_mask = (case_var > 1e-6) | (ctrl_var > 1e-6)
        log(f"  -> 过滤掉 {len(df) - np.sum(valid_genes_mask)} 个无变化基因")
        
        df = df[valid_genes_mask]
        case_vals = case_vals[valid_genes_mask]
        ctrl_vals = ctrl_vals[valid_genes_mask]
        
        log2fc = np.mean(case_vals, axis=1) - np.mean(ctrl_vals, axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tstat, pvals = stats.ttest_ind(case_vals, ctrl_vals, axis=1, equal_var=False)
        
        res_df = pd.DataFrame({
            "gene": df.index,
            "log2fc": log2fc,
            "pval": np.nan_to_num(pvals, nan=1.0)
        })
        
        res_df = res_df.dropna(subset=["log2fc"])
        res_df["padj"] = multipletests(res_df["pval"], method="fdr_bh")[1]
        res_df = res_df.sort_values("log2fc", key=abs, ascending=False)
        
        res_df["gene_symbol"] = res_df["gene"].apply(lambda x: str(x).split("//")[0].split(".")[0].strip().upper())
        
        log(f"✅ 分析成功! 得到 {len(res_df)} 个结果行。")
        return res_df, "Success", res_df, logs

    except Exception as e:
        import traceback
        log(f"❌ 发生未捕获异常: {str(e)}")
        log(traceback.format_exc())
        return None, f"Runtime Error: {str(e)}", None, logs

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
    default_case = "mutation, mutant, variant, patient, knockout, knockdown, disease, clcn, cf, cystic fibrosis, tumor, cancer, treated, stimulation, infected"
    default_ctrl = "control, wt, wild type, wild-type, healthy, normal, vehicle, pbs, dmso, mock, baseline, untreated, placebo, non-targeting"
    case_input = st.text_area("Case 关键词", default_case, height=120)
    ctrl_input = st.text_area("Control 关键词", default_ctrl, height=120)
    case_terms = [x.strip().lower() for x in case_input.split(",") if x.strip()]
    ctrl_terms = [x.strip().lower() for x in ctrl_input.split(",") if x.strip()]
    st.divider()
    top_n_genes = st.number_input("Top Genes", 50, 500, 150)
    taxon_filter = st.selectbox("Species", ["Homo sapiens", "Mus musculus", "All"], index=0)

tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ 搜索", "2️⃣ 样本调试", "3️⃣ 运行分析", "4️⃣ 结果"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1: query_text = st.text_input("Query", value='(CLCN2 OR "chloride channel 2") AND (mutation OR knockout) AND "RNA-seq"')
    with col2: 
        if st.button("Search"):
            with st.spinner("Search..."):
                df = geo_search(query_text)
                if not df.empty and taxon_filter != "All": df = df[df["Taxon"] == taxon_filter]
                st.session_state["geo_hits"] = df
    if not st.session_state["geo_hits"].empty:
        hits = st.session_state["geo_hits"].copy()
        hits.insert(0, "Select", False)
        edited = st.data_editor(hits, column_config={"Select": st.column_config.CheckboxColumn(required=True)}, disabled=["Accession"], use_container_width=True, hide_index=True)
        st.session_state["selected_gses"] = edited[edited["Select"]]["Accession"].tolist()

with tab2:
    if st.session_state["selected_gses"]:
        inspect_gse = st.selectbox("Select Dataset:", st.session_state["selected_gses"])
        if st.button("Load Meta"):
            df_meta, msg = extract_metadata_only(inspect_gse)
            if df_meta is not None: st.session_state["metadata_cache"][inspect_gse] = df_meta
        if inspect_gse in st.session_state["metadata_cache"]:
            df_display = st.session_state["metadata_cache"][inspect_gse].copy()
            df_display["Group"] = df_display["Full_Description"].apply(lambda x: determine_group(x, case_terms, ctrl_terms)[0])
            st.dataframe(df_display, use_container_width=True)

with tab3:
    if st.button("🚀 启动 (Start)", type="primary"):
        results_bucket = []
        log_container = st.container()
        progress = st.progress(0)
        
        for i, gse in enumerate(st.session_state["selected_gses"]):
            with log_container:
                st.write(f"**Processing {gse}...**")
                # 使用诊断版 pipeline
                df_de, msg, full_res, logs = run_analysis_pipeline_diagnostic(gse, case_terms, ctrl_terms)
                
                if df_de is None:
                    st.error(f"❌ {gse} Failed: {msg}")
                    # === 关键：显示详细日志 ===
                    with st.expander("🔍 查看详细报错日志 (Debug Log)", expanded=True):
                        st.text("\n".join(logs))
                    continue
                
                st.success(f"✅ {gse} OK")
                up = df_de[df_de["log2fc"] > 0].head(top_n_genes)["gene_symbol"].tolist()
                dn = df_de[df_de["log2fc"] < 0].tail(top_n_genes)["gene_symbol"].tolist()
                
                if len(up)<5:
                    st.warning("Not enough genes")
                    continue
                    
                df_l = run_l1000fwd(clean_gene_list(up), clean_gene_list(dn))
                df_e = run_enrichr(clean_gene_list(up), "LINCS_L1000_Chem_Pert_down")
                comb = pd.concat([df_l, df_e])
                if not comb.empty:
                    comb["gse"] = gse
                    results_bucket.append(comb)
            progress.progress((i+1)/len(st.session_state["selected_gses"]))
        
        if results_bucket:
            st.session_state["final_drug_rank"] = pd.concat(results_bucket)
            st.success("Done! See Tab 4")

with tab4:
    res = st.session_state.get("final_drug_rank", pd.DataFrame())
    if not res.empty:
        agg = res.groupby("drug").agg(Count=('gse','nunique'), Score=('score','sum')).reset_index().sort_values("Count", ascending=False)
        st.dataframe(agg, use_container_width=True)
        st.download_button("Download", agg.to_csv().encode("utf-8"), "drugs.csv")
