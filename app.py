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
from bs4 import BeautifulSoup 

# ==========================================
# 0. 配置与初始化
# ==========================================
st.set_page_config(page_title="GEO Pipeline (Final Fix)", layout="wide", page_icon="🧬")

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
        r = requests.get(url, stream=True, timeout=120)
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
    suppl_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/suppl/"
    return soft_url, matrix_url, suppl_url

def extract_metadata_only(gse):
    gse_dir = RAW_DIR / gse
    gse_dir.mkdir(exist_ok=True)
    soft_url, _, _ = get_geo_urls(gse)
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

# --- 补充文件处理模块 (Enhanced) ---

def find_best_suppl_file(suppl_url):
    try:
        r = requests.get(suppl_url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        files = [a.get('href') for a in soup.find_all('a') if a.get('href')]
        candidates = []
        for f in files:
            f_lower = f.lower()
            if f_lower.endswith(('.txt.gz', '.tsv.gz', '.csv.gz', '.xls', '.xlsx', '.txt', '.tsv')):
                score = 0
                if 'count' in f_lower: score += 3
                if 'fpkm' in f_lower: score += 2
                if 'tpm' in f_lower: score += 2
                if 'expression' in f_lower: score += 2
                if 'raw' in f_lower: score += 1
                candidates.append((score, f))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1] if candidates else None
    except: return None

def normalize_str_token(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def try_map_suppl_cols(df, df_meta, logs):
    """
    尝试三种策略映射列名：
    1. GSM 直接匹配
    2. Title 模糊匹配
    3. 强制位置匹配 (如果样本数一致)
    """
    
    # 打印前几个列名供调试
    logs.append(f"  -> 补充文件列名预览 (前5): {list(df.columns[:5])}")
    logs.append(f"  -> Metadata GSM顺序预览: {df_meta['GSM'].tolist()[:5]}")
    
    # 策略 1: GSM 匹配
    new_cols = {}
    for col in df.columns:
        m = re.search(r'(GSM\d+)', str(col))
        if m: new_cols[col] = m.group(1)
    
    if len(new_cols) >= 2:
        logs.append(f"  -> 策略1成功: 匹配到 {len(new_cols)} 个 GSM")
        df = df.rename(columns=new_cols)
        return df[[c for c in df.columns if c.startswith("GSM")]], "GSM Match"

    # 策略 2: Title 模糊匹配
    title_to_gsm = {normalize_str_token(row["Title"]): row["GSM"] for _, row in df_meta.iterrows()}
    new_cols = {}
    for col in df.columns:
        norm_col = normalize_str_token(col)
        for t_norm, gsm in title_to_gsm.items():
            if t_norm in norm_col or norm_col in t_norm:
                if len(t_norm) > 3: # 避免匹配到纯数字
                    new_cols[col] = gsm
                    break
    
    if len(new_cols) >= 2:
        logs.append(f"  -> 策略2成功: Title模糊匹配到 {len(new_cols)} 个")
        df = df.rename(columns=new_cols)
        # 去重，防止多个列匹配到同一个GSM
        df = df.loc[:, ~df.columns.duplicated()]
        return df[list(new_cols.values())], "Title Fuzzy Match"

    # 策略 3: 强制位置匹配 (Positional Fallback)
    # 找出 df 中看起来像数据的列（排除 gene id 等）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 如果没识别出数字列，可能是还没转数字，取所有列排除第一列
    if len(numeric_cols) < 2:
        numeric_cols = df.columns[1:].tolist()
    
    meta_gsms = df_meta["GSM"].tolist()
    
    logs.append(f"  -> 策略3检查: Meta样本数={len(meta_gsms)}, Matrix数据列数={len(numeric_cols)}")
    
    if len(meta_gsms) == len(numeric_cols):
        logs.append("  -> ⚠️ 启用策略3: 强制按位置对齐 (Positional Mapping)")
        # 直接按顺序重命名
        rename_map = {old: new for old, new in zip(numeric_cols, meta_gsms)}
        df = df.rename(columns=rename_map)
        return df[meta_gsms], "Positional Force Match"
    
    return df, "Mapping Failed"

# --- 深度诊断版 Pipeline ---

def run_analysis_pipeline_diagnostic(gse, case_terms, ctrl_terms):
    logs = []
    def log(msg): logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    
    log(f"=== 开始处理 {gse} ===")
    gse_dir = RAW_DIR / gse
    gse_dir.mkdir(exist_ok=True)
    soft_url, matrix_url, suppl_url = get_geo_urls(gse)
    soft_path = gse_dir / f"{gse}_family.soft.gz"
    matrix_path = gse_dir / f"{gse}_series_matrix.txt.gz"
    
    # 1. 下载 Soft
    try: download_file(soft_url, soft_path)
    except: return None, "Soft文件下载失败", None, logs

    # 2. 解析元数据
    df_meta, msg = extract_metadata_only(gse)
    if df_meta is None or df_meta.empty: return None, "Soft解析失败", None, logs
    
    conditions = {}
    for idx, row in df_meta.iterrows():
        clean_gsm = str(row["GSM"]).strip()
        group, _ = determine_group(row["Full_Description"], case_terms, ctrl_terms)
        if group in ["Case", "Control"]: conditions[clean_gsm] = group.lower()
    
    if not conditions: return None, "没有样本匹配到关键词", None, logs
    log(f"元数据分组: {len(conditions)} 个样本 (Case/Ctrl)")

    # 3. 尝试读取标准矩阵
    use_suppl = False
    df = pd.DataFrame()
    
    log("尝试下载标准 Series Matrix...")
    try:
        download_file(matrix_url, matrix_path)
        header_row = None
        with gzip.open(matrix_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i>2000: break
                if "!series_matrix_table_begin" in line or "\"ID_REF\"" in line or "ID_REF" in line:
                    header_row = i if "ID_REF" in line else i+1
                    break
        skip = header_row if header_row is not None else "infer"
        if skip == "infer": df = pd.read_csv(matrix_path, sep="\t", comment="!", index_col=0, on_bad_lines='skip')
        else: df = pd.read_csv(matrix_path, sep="\t", skiprows=skip, index_col=0, on_bad_lines='skip')
        
        log(f"标准矩阵形状: {df.shape}")
        if df.shape[0] < 50 or df.shape[1] < 2:
            log("⚠️ 检测到矩阵为空或行数过少 (可能是RNA-seq空壳文件)。")
            use_suppl = True
        else:
            clean_map = {}
            for c in df.columns:
                m = re.search(r'(GSM\d+)', str(c))
                if m: clean_map[c] = m.group(1)
            if clean_map:
                df = df.rename(columns=clean_map)
                df = df.loc[:, ~df.columns.duplicated()]
            else:
                log("⚠️ 标准矩阵列名不含 GSM，标记为无效。")
                use_suppl = True
    except Exception as e:
        log(f"标准矩阵读取出错: {e}")
        use_suppl = True

    # 4. 补充文件流程
    if use_suppl:
        log("🔄 启动备用方案: 抓取 Supplementary Files...")
        best_file = find_best_suppl_file(suppl_url)
        if best_file:
            log(f"找到补充文件: {best_file}")
            suppl_path = gse_dir / best_file
            file_url = suppl_url + best_file
            try:
                download_file(file_url, suppl_path)
                # 尝试自动检测分隔符和压缩格式
                if best_file.endswith('.csv.gz') or best_file.endswith('.csv'):
                    df = pd.read_csv(suppl_path, index_col=0)
                else:
                    df = pd.read_csv(suppl_path, sep=None, engine='python', index_col=0)
                
                log(f"补充文件读取成功，形状: {df.shape}")
                
                # === 关键修复：列名映射 + 强制对齐 ===
                df, map_msg = try_map_suppl_cols(df, df_meta, logs)
                log(f"列名映射结果: {map_msg}")
                
            except Exception as e:
                log(f"❌ 补充文件读取失败: {e}")
                return None, "补充文件读取失败", None, logs
        else:
            log("❌ 未找到合适的补充文件。")
            return None, "无有效矩阵文件", None, logs

    # 5. 对齐
    common = set(df.columns).intersection(set(conditions.keys()))
    log(f"最终对齐样本数: {len(common)}")
    
    if len(common) < 2:
        return None, "样本对齐失败 (Matrix列名与Meta不匹配)", None, logs
        
    df = df[list(common)]
    
    # 6. 转数值
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=0, how='all')
    if df.empty: return None, "矩阵数值化后为空", None, logs
    
    # Log2
    if df.max().max() > 50:
        log("执行 Log2 转换...")
        df = np.log2(df + 1)
    
    # 7. 差异分析
    case_cols = [c for c in df.columns if conditions[c] == 'case']
    ctrl_cols = [c for c in df.columns if conditions[c] == 'control']
    
    if len(case_cols) < 2 or len(ctrl_cols) < 2:
        return None, f"有效样本不足 (Case={len(case_cols)}, Ctrl={len(ctrl_cols)})", None, logs
        
    case_vals = df[case_cols].values
    ctrl_vals = df[ctrl_cols].values
    mask = (np.mean(case_vals, axis=1) > 0) | (np.mean(ctrl_vals, axis=1) > 0)
    df = df[mask]
    case_vals = case_vals[mask]
    ctrl_vals = ctrl_vals[mask]
    
    log2fc = np.mean(case_vals, axis=1) - np.mean(ctrl_vals, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        tstat, pvals = stats.ttest_ind(case_vals, ctrl_vals, axis=1, equal_var=False)
    
    res_df = pd.DataFrame({"gene": df.index, "log2fc": log2fc, "pval": np.nan_to_num(pvals, nan=1.0)})
    res_df = res_df.dropna(subset=["log2fc"])
    res_df["padj"] = multipletests(res_df["pval"], method="fdr_bh")[1]
    res_df = res_df.sort_values("log2fc", key=abs, ascending=False)
    res_df["gene_symbol"] = res_df["gene"].apply(lambda x: str(x).split("//")[0].split(".")[0].strip().upper())
    
    log(f"✅ 分析完成，得到 {len(res_df)} 个基因。")
    return res_df, "Success", res_df, logs

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
                df_de, msg, full_res, logs = run_analysis_pipeline_diagnostic(gse, case_terms, ctrl_terms)
                
                if df_de is None:
                    st.error(f"❌ {gse} Failed: {msg}")
                    with st.expander("🔍 查看详细报错日志 (Debug Log)", expanded=True):
                        st.text("\n".join(logs))
                    continue
                
                st.success(f"✅ {gse} OK. Genes: {len(df_de)}")
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
