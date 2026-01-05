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
st.set_page_config(page_title="GEO Pipeline (Final & ID Fixed)", layout="wide", page_icon="💊")

WORK_DIR = Path("workspace")
RAW_DIR = WORK_DIR / "raw"
PROC_DIR = WORK_DIR / "proc"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

if "geo_hits" not in st.session_state: st.session_state["geo_hits"] = pd.DataFrame()
if "selected_gses" not in st.session_state: st.session_state["selected_gses"] = []
if "metadata_cache" not in st.session_state: st.session_state["metadata_cache"] = {}
if "de_results_cache" not in st.session_state: st.session_state["de_results_cache"] = {}
if "final_drug_rank" not in st.session_state: st.session_state["final_drug_rank"] = pd.DataFrame()

# ==========================================
# 1. 核心工具函数
# ==========================================

@st.cache_resource
def get_mygene_info(): return mygene.MyGeneInfo()

# --- 核心修复：ID 转换函数 ---
def map_ensembl_to_symbol(gene_list, logs):
    """
    将 ENSG ID 转换为 Symbol (例如 ENSG00000182378 -> CLCN2)
    """
    # 检查是否看起来像 Ensembl ID
    sample = [str(g) for g in gene_list[:10]]
    if not any(x.startswith("ENS") for x in sample):
        return {g: g for g in gene_list} # 已经是 Symbol，无需转换

    logs.append("  -> 检测到 Ensembl ID，正在调用 MyGene 进行转换...")
    mg = get_mygene_info()
    try:
        # 批量查询
        res = mg.querymany(gene_list, scopes='ensembl.gene', fields='symbol', species='human', returnall=False, verbose=False)
        mapping = {}
        success_count = 0
        for item in res:
            query = item.get('query')
            symbol = item.get('symbol')
            if query and symbol:
                mapping[query] = symbol.upper()
                success_count += 1
        
        logs.append(f"  -> ID 转换成功: {success_count}/{len(gene_list)}")
        return mapping
    except Exception as e:
        logs.append(f"  -> MyGene ID 转换失败: {e}")
        return {g: g for g in gene_list}

def clean_gene_list(genes):
    out = []
    seen = set()
    for g in genes:
        if not isinstance(g, str): continue
        # 这里不再暴力去版本号，保留完整字符交给 mapping 处理
        g = g.strip().upper()
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

# --- 补充文件处理 ---

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
                if 'deg' in f_lower: score += 4
                if 'result' in f_lower: score += 3
                if 'count' in f_lower: score += 3
                if 'fpkm' in f_lower: score += 2
                if 'tpm' in f_lower: score += 2
                candidates.append((score, f))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1] if candidates else None
    except: return None

def normalize_str_token(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def try_map_suppl_cols(df, df_meta, logs):
    logs.append(f"  -> 补充文件列名预览 (前5): {list(df.columns[:5])}")
    col_str = " ".join([str(c).lower() for c in df.columns])
    if "foldchange" in col_str or "logfc" in col_str or "pvalue" in col_str or "padj" in col_str:
        logs.append("  -> ✅ 检测到文件包含差异分析结果字段 (logFC/Pval)，启用直读模式。")
        return df, "Pre-calculated DE"

    new_cols = {}
    for col in df.columns:
        m = re.search(r'(GSM\d+)', str(col))
        if m: new_cols[col] = m.group(1)
    if len(new_cols) >= 2:
        logs.append(f"  -> 策略1成功: 匹配到 {len(new_cols)} 个 GSM")
        df = df.rename(columns=new_cols)
        return df[[c for c in df.columns if c.startswith("GSM")]], "GSM Match"

    title_to_gsm = {normalize_str_token(row["Title"]): row["GSM"] for _, row in df_meta.iterrows()}
    new_cols = {}
    for col in df.columns:
        norm_col = normalize_str_token(col)
        for t_norm, gsm in title_to_gsm.items():
            if t_norm in norm_col or norm_col in t_norm:
                if len(t_norm) > 3:
                    new_cols[col] = gsm
                    break
    if len(new_cols) >= 2:
        logs.append(f"  -> 策略2成功: Title模糊匹配到 {len(new_cols)} 个")
        df = df.rename(columns=new_cols)
        df = df.loc[:, ~df.columns.duplicated()]
        return df[list(new_cols.values())], "Title Fuzzy Match"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2: numeric_cols = df.columns[1:].tolist()
    meta_gsms = df_meta["GSM"].tolist()
    
    if len(meta_gsms) == len(numeric_cols):
        logs.append("  -> ⚠️ 启用策略3: 强制按位置对齐 (Positional Mapping)")
        rename_map = {old: new for old, new in zip(numeric_cols, meta_gsms)}
        df = df.rename(columns=rename_map)
        return df[meta_gsms], "Positional Force Match"
    
    return df, "Mapping Failed"

# --- Pipeline ---

def run_analysis_pipeline_diagnostic(gse, case_terms, ctrl_terms):
    logs = []
    def log(msg): logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    
    log(f"=== 开始处理 {gse} ===")
    gse_dir = RAW_DIR / gse
    gse_dir.mkdir(exist_ok=True)
    soft_url, matrix_url, suppl_url = get_geo_urls(gse)
    soft_path = gse_dir / f"{gse}_family.soft.gz"
    matrix_path = gse_dir / f"{gse}_series_matrix.txt.gz"
    
    try: download_file(soft_url, soft_path)
    except: return None, "Soft下载失败", None, logs

    df_meta, msg = extract_metadata_only(gse)
    if df_meta is None or df_meta.empty: return None, "Soft解析失败", None, logs
    
    conditions = {}
    for idx, row in df_meta.iterrows():
        clean_gsm = str(row["GSM"]).strip()
        group, _ = determine_group(row["Full_Description"], case_terms, ctrl_terms)
        if group in ["Case", "Control"]: conditions[clean_gsm] = group.lower()
    
    if not conditions: return None, "无样本匹配到关键词", None, logs

    use_suppl = False
    df = pd.DataFrame()
    is_precalculated = False
    
    log("尝试下载标准矩阵...")
    try:
        download_file(matrix_url, matrix_path)
        df = pd.read_csv(matrix_path, sep="\t", comment="!", index_col=0, on_bad_lines='skip')
        if df.shape[0] < 50 or df.shape[1] < 2:
            log("⚠️ 标准矩阵无效 (空壳)。")
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
                use_suppl = True
    except:
        use_suppl = True

    if use_suppl:
        log("🔄 抓取补充文件...")
        best_file = find_best_suppl_file(suppl_url)
        if best_file:
            log(f"下载补充文件: {best_file}")
            suppl_path = gse_dir / best_file
            try:
                download_file(suppl_url + best_file, suppl_path)
                if best_file.endswith('.csv.gz') or best_file.endswith('.csv'):
                    df = pd.read_csv(suppl_path, index_col=0)
                else:
                    df = pd.read_csv(suppl_path, sep=None, engine='python', index_col=0)
                
                log(f"读取成功: {df.shape}")
                df, map_msg = try_map_suppl_cols(df, df_meta, logs)
                log(f"映射状态: {map_msg}")
                if map_msg == "Pre-calculated DE":
                    is_precalculated = True
            except Exception as e:
                return None, f"补充文件错误: {e}", None, logs
        else:
            return None, "无有效文件", None, logs

    # === 生成 res_df ===
    res_df = pd.DataFrame()

    if is_precalculated:
        log("直读模式...")
        cols = df.columns
        lfc_col = next((c for c in cols if "log2foldchange" in c.lower() or "logfc" in c.lower()), None)
        pval_col = next((c for c in cols if "adj.p" in c.lower() or "padj" in c.lower() or "pvalue" in c.lower()), None)
        
        if lfc_col:
            res_df["gene"] = df.index
            res_df["log2fc"] = df[lfc_col]
            if pval_col:
                res_df["padj"] = df[pval_col]
                res_df["pval"] = df[pval_col]
            else:
                res_df["pval"] = 0.05
                res_df["padj"] = 0.05
        else:
            return None, "未找到 LogFC 列", None, logs
    else:
        # 常规计算
        common = set(df.columns).intersection(set(conditions.keys()))
        if len(common) < 2: return None, "样本对齐失败", None, logs
        
        df = df[list(common)]
        df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='all')
        if df.empty: return None, "矩阵为空", None, logs
        if df.max().max() > 50: df = np.log2(df + 1)
            
        case_cols = [c for c in df.columns if conditions[c] == 'case']
        ctrl_cols = [c for c in df.columns if conditions[c] == 'control']
        
        if len(case_cols)<2 or len(ctrl_cols)<2: return None, "样本不足", None, logs
        
        case_vals = df[case_cols].values
        ctrl_vals = df[ctrl_cols].values
        
        log2fc = np.mean(case_vals, axis=1) - np.mean(ctrl_vals, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            _, pvals = stats.ttest_ind(case_vals, ctrl_vals, axis=1, equal_var=False)
        
        res_df["gene"] = df.index
        res_df["log2fc"] = log2fc
        res_df["pval"] = np.nan_to_num(pvals, nan=1.0)
        res_df["padj"] = multipletests(res_df["pval"], method="fdr_bh")[1]

    # === 关键修复：ID 转换 ===
    res_df = res_df.dropna(subset=["log2fc"])
    res_df = res_df.sort_values("log2fc", key=abs, ascending=False)
    
    # 1. 先简单清洗（去版本号）
    raw_genes = res_df["gene"].astype(str).apply(lambda x: x.split("//")[0].split(".")[0].strip())
    
    # 2. 调用 mapping (Ensembl -> Symbol)
    mapping_dict = map_ensembl_to_symbol(raw_genes.tolist(), logs)
    res_df["gene_symbol"] = raw_genes.map(mapping_dict).fillna(raw_genes).str.upper()
    
    # 3. 过滤无效 Symbol
    res_df = res_df[res_df["gene_symbol"] != ""]
    
    log(f"✅ 完成。Symbol 转换后剩余 {len(res_df)} 个基因。")
    return res_df, "Success", res_df, logs

# --- API ---
def run_l1000fwd(up_genes, dn_genes, logs):
    url = "https://maayanlab.cloud/l1000fwd/sig_search"
    # L1000 只接受 Symbol，且有长度限制
    payload = {"up_genes": up_genes[:100], "down_genes": dn_genes[:100]}
    
    # 简单的重试机制
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=45)
            if r.status_code == 200:
                res_id = r.json().get("result_id")
                if not res_id: return pd.DataFrame()
                time.sleep(2)
                r2 = requests.get(f"https://maayanlab.cloud/l1000fwd/result/topn/{res_id}", timeout=45)
                data = r2.json()
                rows = []
                if "opposite" in data:
                    for item in data["opposite"]:
                        rows.append({"drug": item.get("pert_id"), "score": item.get("score"), "source": "L1000FWD"})
                return pd.DataFrame(rows)
            else:
                logs.append(f"L1000 Attempt {attempt+1} failed: {r.status_code}")
                time.sleep(2)
        except Exception as e:
            logs.append(f"L1000 Error: {e}")
            time.sleep(2)
    return pd.DataFrame()

def run_enrichr(genes, logs, library="LINCS_L1000_Chem_Pert_down"):
    base = "https://maayanlab.cloud/Enrichr"
    try:
        r = requests.post(f"{base}/addList", files={'list': (None, '\n'.join(genes[:300])), 'description': (None, 'Streamlit')}, timeout=45)
        uid = r.json().get("userListId")
        if not uid: return pd.DataFrame()
        r2 = requests.get(f"{base}/enrich?userListId={uid}&backgroundType={library}", timeout=45)
        data = r2.json()
        if library not in data: return pd.DataFrame()
        rows = []
        for item in data[library]:
            rows.append({"drug": item[1].split("_")[0].split(" ")[0], "score": item[4], "source": "Enrichr"})
        return pd.DataFrame(rows)
    except Exception as e:
        logs.append(f"Enrichr Error: {e}")
        return pd.DataFrame()

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
    top_n_genes = st.number_input("Top Genes (Signature)", 50, 500, 100)
    taxon_filter = st.selectbox("Species", ["Homo sapiens", "Mus musculus", "All"], index=0)

tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ 搜索", "2️⃣ 样本调试", "3️⃣ 运行分析", "4️⃣ 结果 & 详情"])

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
                    with st.expander("Debug Log"): st.text("\n".join(logs))
                    continue
                
                st.session_state["de_results_cache"][gse] = full_res
                
                st.success(f"✅ {gse} OK. Genes: {len(df_de)}")
                
                # 使用转换后的 Symbol 列
                up = df_de[df_de["log2fc"] > 0].head(top_n_genes)["gene_symbol"].tolist()
                dn = df_de[df_de["log2fc"] < 0].tail(top_n_genes)["gene_symbol"].tolist()
                
                # 过滤掉空的 Symbol
                up = [x for x in up if x and not x.startswith("ENS")]
                dn = [x for x in dn if x and not x.startswith("ENS")]
                
                if len(up)<10:
                    st.warning("Skipping drugs (too few valid symbols)")
                    continue
                
                logs.append("正在查询 API (Sent Symbols)...")
                df_l = run_l1000fwd(up, dn, logs)
                df_e = run_enrichr(up, logs, "LINCS_L1000_Chem_Pert_down")
                
                comb = pd.concat([df_l, df_e])
                if not comb.empty:
                    comb["gse"] = gse
                    results_bucket.append(comb)
                else:
                    st.warning("API返回空结果 (Check Log)")
                    with st.expander("Debug Log"): st.text("\n".join(logs))
                    
            progress.progress((i+1)/len(st.session_state["selected_gses"]))
        
        if results_bucket:
            st.session_state["final_drug_rank"] = pd.concat(results_bucket)
            st.success("Done! See Tab 4")

with tab4:
    st.subheader("💊 药物分析结果")
    res = st.session_state.get("final_drug_rank", pd.DataFrame())
    
    col_drug, col_de = st.columns([1, 1])
    
    with col_drug:
        st.markdown("#### 1. 药物重定位排序")
        if not res.empty:
            agg = res.groupby("drug").agg(Count=('gse','nunique'), Score=('score','sum')).reset_index().sort_values("Count", ascending=False)
            st.dataframe(agg, use_container_width=True, height=500)
            st.download_button("Download Drugs", agg.to_csv().encode("utf-8"), "drugs.csv")
        else:
            st.info("暂无药物结果")

    with col_de:
        st.markdown("#### 2. 差异表达详细数据 (查看 pval, logFC)")
        if st.session_state["de_results_cache"]:
            view_gse = st.selectbox("选择数据集查看:", list(st.session_state["de_results_cache"].keys()))
            de_df = st.session_state["de_results_cache"][view_gse]
            st.dataframe(de_df.head(100), use_container_width=True, height=500)
            st.download_button(f"Download {view_gse} DE Table", de_df.to_csv().encode("utf-8"), f"{view_gse}_DE.csv")
        else:
            st.info("暂无差异分析数据，请先运行 Pipeline")
