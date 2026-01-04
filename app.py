import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import os
import shutil
import time
import json
import gzip
import tarfile
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from bs4 import BeautifulSoup
import mygene
from urllib.parse import quote
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. 配置与初始化
# ==========================================
st.set_page_config(page_title="GEO Drug Repurposing Pipeline", layout="wide")

# 定义工作目录 (使用 Streamlit 的临时目录或本地目录)
WORK_DIR = Path("workspace")
RAW_DIR = WORK_DIR / "raw"
PROC_DIR = WORK_DIR / "proc"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# 缓存配置
st.session_state.setdefault("geo_hits", pd.DataFrame())
st.session_state.setdefault("analysis_results", [])
st.session_state.setdefault("final_drug_rank", pd.DataFrame())

# ==========================================
# 1. 工具函数 (从原脚本精简移植)
# ==========================================

@st.cache_resource
def get_mygene_info():
    return mygene.MyGeneInfo()

def clean_gene_list(genes):
    out = []
    seen = set()
    for g in genes:
        if not isinstance(g, str): continue
        g = g.split(".")[0].strip().upper() # 去除Ensembl版本号
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out

def map_to_symbols(genes, species="human"):
    mg = get_mygene_info()
    genes = clean_gene_list(genes)
    if not genes: return []
    
    # 简单的批量查询
    res = mg.querymany(genes, scopes=["symbol", "ensembl.gene", "entrezgene"], 
                       fields="symbol", species=species, verbose=False, returnall=False)
    
    symbols = []
    for r in res:
        if "symbol" in r:
            symbols.append(r["symbol"].upper())
    return list(set(symbols))

# --- GEO 下载与解析 ---
def geo_search(query, retmax=20):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    # Search
    search_url = f"{base}/esearch.fcgi?db=gds&term={quote(query)}&retmax={retmax}&retmode=json"
    r = requests.get(search_url).json()
    ids = r.get("esearchresult", {}).get("idlist", [])
    if not ids: return pd.DataFrame()
    
    # Summary
    summary_url = f"{base}/esummary.fcgi?db=gds&id={','.join(ids)}&retmode=json"
    r = requests.get(summary_url).json()
    result = r.get("result", {})
    
    rows = []
    for uid in ids:
        if uid not in result: continue
        item = result[uid]
        if not item.get("accession", "").startswith("GSE"): continue
        rows.append({
            "Accession": item.get("accession"),
            "Title": item.get("title"),
            "Summary": item.get("summary", "")[:200] + "...",
            "Taxon": item.get("taxon", ""),
            "Samples": item.get("n_samples"),
            "Date": item.get("pdat")
        })
    return pd.DataFrame(rows)

def download_file(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0: return
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk: f.write(chunk)

def get_geo_urls(gse):
    # 简化的 URL 生成
    gse = gse.upper()
    prefix = gse[:-3] + "nnn"
    soft_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/soft/{gse}_family.soft.gz"
    matrix_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/matrix/{gse}_series_matrix.txt.gz"
    return soft_url, matrix_url

def parse_soft_conditions(soft_path, case_terms, ctrl_terms):
    # 极简版 Soft 解析与分组
    meta = {}
    current_gsm = None
    
    with gzip.open(soft_path, 'rt', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith("^SAMPLE ="):
                current_gsm = line.split("=")[1].strip()
                meta[current_gsm] = ""
            elif current_gsm and (line.startswith("!Sample_title") or line.startswith("!Sample_source_name")):
                meta[current_gsm] += " " + line.split("=")[1].strip().lower()
                
    conditions = {}
    for gsm, text in meta.items():
        is_case = any(t in text for t in case_terms)
        is_ctrl = any(t in text for t in ctrl_terms)
        
        if is_case and not is_ctrl: conditions[gsm] = "case"
        elif is_ctrl and not is_case: conditions[gsm] = "control"
        else: conditions[gsm] = "ambiguous"
        
    return pd.Series(conditions)

# --- 差异分析 ---
def run_ttest_pipeline(gse, case_terms, ctrl_terms):
    gse_dir = RAW_DIR / gse
    gse_dir.mkdir(exist_ok=True)
    
    soft_url, matrix_url = get_geo_urls(gse)
    soft_path = gse_dir / f"{gse}_family.soft.gz"
    matrix_path = gse_dir / f"{gse}_series_matrix.txt.gz"
    
    # 1. Download
    download_file(soft_url, soft_path)
    download_file(matrix_url, matrix_path)
    
    # 2. Parse Conditions
    conditions = parse_soft_conditions(soft_path, case_terms, ctrl_terms)
    case_samples = conditions[conditions == "case"].index.tolist()
    ctrl_samples = conditions[conditions == "control"].index.tolist()
    
    if len(case_samples) < 2 or len(ctrl_samples) < 2:
        return None, f"Insufficient samples: Case={len(case_samples)}, Ctrl={len(ctrl_samples)}"
    
    # 3. Load Matrix
    try:
        df = pd.read_csv(matrix_path, sep="\t", comment="!", index_col=0)
        # 简单清洗：去除空值，取对数(如果值很大)
        df = df.dropna()
        if df.max().max() > 50:
            df = np.log2(df + 1)
    except Exception as e:
        return None, f"Matrix parse error: {str(e)}"
    
    # 对齐样本
    valid_cols = [c for c in df.columns if c in conditions.index]
    df = df[valid_cols]
    
    case_cols = [c for c in df.columns if conditions.get(c) == "case"]
    ctrl_cols = [c for c in df.columns if conditions.get(c) == "control"]
    
    # 4. T-test
    results = []
    for gene, row in df.iterrows():
        case_vals = row[case_cols].values
        ctrl_vals = row[ctrl_cols].values
        if len(case_vals) < 2 or len(ctrl_vals) < 2: continue
        
        t, p = stats.ttest_ind(case_vals, ctrl_vals, equal_var=False)
        lfc = np.mean(case_vals) - np.mean(ctrl_vals)
        results.append({"gene": gene, "log2fc": lfc, "pval": p})
        
    res_df = pd.DataFrame(results).dropna()
    if res_df.empty: return None, "No valid DE results"
    
    # FDR
    res_df["padj"] = multipletests(res_df["pval"], method="fdr_bh")[1]
    res_df = res_df.sort_values("padj")
    
    # Map to Symbols (Simplified: assume index is roughly Symbol or needs mapping)
    # 实际场景可能需要 ID mapping，这里假设 Matrix 主要是 Symbol 或能被处理
    # 为了演示，我们只取前缀
    res_df["gene_symbol"] = res_df["gene"].apply(lambda x: str(x).split("//")[0].strip()) 
    
    return res_df, f"Success: Case={len(case_cols)}, Ctrl={len(ctrl_cols)}"

# --- Connectivity APIs ---

def run_l1000fwd(up_genes, dn_genes):
    url = "https://maayanlab.cloud/l1000fwd/sig_search"
    payload = {"up_genes": up_genes[:100], "down_genes": dn_genes[:100]}
    try:
        r = requests.post(url, json=payload)
        res_id = r.json().get("result_id")
        if not res_id: return pd.DataFrame()
        
        # Get Top results
        time.sleep(1)
        r2 = requests.get(f"https://maayanlab.cloud/l1000fwd/result/topn/{res_id}")
        data = r2.json()
        
        rows = []
        if "opposite" in data:
            for item in data["opposite"]:
                rows.append({
                    "drug": item.get("pert_id"), # 这里通常需要二次查询名字，暂用 ID
                    "score": item.get("score"),
                    "source": "L1000FWD",
                    "direction": "opposite"
                })
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

def run_enrichr(genes, library="LINCS_L1000_Chem_Pert_down"):
    base = "https://maayanlab.cloud/Enrichr"
    # Add List
    try:
        r = requests.post(f"{base}/addList", files={
            'list': (None, '\n'.join(genes)),
            'description': (None, 'demo')
        })
        user_list_id = r.json().get("userListId")
        
        # Enrich
        r2 = requests.get(f"{base}/enrich?userListId={user_list_id}&backgroundType={library}")
        data = r2.json()
        if library not in data: return pd.DataFrame()
        
        rows = []
        for item in data[library]:
            # item: [rank, term, pval, zscore, combined_score, ...]
            rows.append({
                "drug": item[1].split("_")[0], # 简单提取药名
                "score": item[4],
                "pval": item[2],
                "source": "Enrichr"
            })
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

# ==========================================
# 2. Streamlit 界面逻辑
# ==========================================

st.title("💊 自动化药物重定位分析平台 (Pipeline v5.2)")
st.markdown("基于 GEO 转录组数据 -> 差异表达 -> L1000FWD/Enrichr -> 药物推荐")

# --- Sidebar: 设置 ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    taxon_filter = st.selectbox("物种过滤", ["Homo sapiens", "Mus musculus", "All"], index=0)
    st.divider()
    
    st.subheader("分组推断关键词")
    default_case = "mutation, mutant, variant, patient, knockout, knockdown, disease, clcn"
    default_ctrl = "control, wt, wild type, healthy, normal, vehicle"
    
    case_input = st.text_area("实验组 (Case) 关键词", default_case)
    ctrl_input = st.text_area("对照组 (Control) 关键词", default_ctrl)
    
    case_terms = [x.strip().lower() for x in case_input.split(",")]
    ctrl_terms = [x.strip().lower() for x in ctrl_input.split(",")]
    
    st.divider()
    top_n_genes = st.number_input("Signature 基因数量", 50, 500, 150)

# --- Tab 1: 搜索与选择 ---
tab1, tab2, tab3 = st.tabs(["1. 搜索 GEO 数据", "2. 运行分析流程", "3. 药物排序结果"])

with tab1:
    st.subheader("🔍 搜索 GEO 数据集")
    col1, col2 = st.columns([3, 1])
    with col1:
        query_text = st.text_input("输入查询 (例如: CLCN2 mutation RNA-seq)", 
                                   value='(CLCN2 OR "chloride channel 2") AND (mutation OR knockout) AND "RNA-seq"')
    with col2:
        search_btn = st.button("开始搜索", use_container_width=True)
    
    if search_btn and query_text:
        with st.spinner("正在搜索 NCBI GEO..."):
            df_hits = geo_search(query_text, retmax=50)
            if not df_hits.empty:
                if taxon_filter != "All":
                    df_hits = df_hits[df_hits["Taxon"] == taxon_filter]
                st.session_state["geo_hits"] = df_hits
            else:
                st.warning("未找到相关数据集，请尝试放宽搜索条件。")
    
    if not st.session_state["geo_hits"].empty:
        st.write(f"找到 {len(st.session_state['geo_hits'])} 个数据集:")
        
        # 使用 DataEditor 让用户勾选
        hits_display = st.session_state["geo_hits"].copy()
        hits_display["Select"] = False
        edited_df = st.data_editor(hits_display, 
                                   column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                                   disabled=["Accession", "Title", "Summary"],
                                   use_container_width=True)
        
        selected_gses = edited_df[edited_df["Select"]]["Accession"].tolist()
        st.session_state["selected_gses"] = selected_gses
        st.info(f"已选择 {len(selected_gses)} 个 GSE 进行分析: {', '.join(selected_gses)}")
    else:
        st.write("暂无搜索结果。")

# --- Tab 2: 运行流程 ---
with tab2:
    st.subheader("⚡ 批处理分析")
    
    if "selected_gses" not in st.session_state or not st.session_state["selected_gses"]:
        st.warning("请先在第 1 步选择数据集。")
    else:
        if st.button("🚀 启动分析 Pipeline", type="primary"):
            results_bucket = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_gse = len(st.session_state["selected_gses"])
            
            for i, gse in enumerate(st.session_state["selected_gses"]):
                status_text.text(f"正在处理 {gse} ({i+1}/{total_gse})... 下载 & 差异分析")
                
                # 1. 下载 & 差异分析
                df_de, msg = run_ttest_pipeline(gse, case_terms, ctrl_terms)
                
                if df_de is None:
                    st.error(f"❌ {gse}: 失败 - {msg}")
                    continue
                
                st.success(f"✅ {gse}: 差异分析完成 ({msg})")
                
                # 2. 提取 Signature
                up_genes = df_de.sort_values("log2fc", ascending=False).head(top_n_genes)["gene_symbol"].tolist()
                dn_genes = df_de.sort_values("log2fc", ascending=True).head(top_n_genes)["gene_symbol"].tolist()
                
                # 清洗基因名
                up_genes = clean_gene_list(up_genes)
                dn_genes = clean_gene_list(dn_genes)
                
                # 3. 药物连接性预测 (API Calls)
                status_text.text(f"正在处理 {gse}... 查询 L1000FWD & Enrichr")
                
                # L1000FWD (寻找反转信号 - Opposite)
                df_l1000 = run_l1000fwd(up_genes, dn_genes)
                
                # Enrichr (Input UP genes vs Drug DOWN lib = Reversal)
                df_enrichr = run_enrichr(up_genes, library="LINCS_L1000_Chem_Pert_down")
                
                # 简单的结果合并
                combined_drugs = pd.concat([
                    df_l1000[["drug", "score", "source"]],
                    df_enrichr[["drug", "score", "source"]]
                ])
                
                if not combined_drugs.empty:
                    combined_drugs["gse"] = gse
                    results_bucket.append(combined_drugs)
                    with st.expander(f"{gse} 初步候选药物 (Top 5)"):
                        st.dataframe(combined_drugs.head(5))
                
                progress_bar.progress((i + 1) / total_gse)
            
            status_text.text("分析完成！正在汇总...")
            
            if results_bucket:
                final_df = pd.concat(results_bucket)
                st.session_state["final_drug_rank"] = final_df
                st.balloons()
            else:
                st.warning("所有数据集均未能产生有效药物结果。")

# --- Tab 3: 结果展示 ---
with tab3:
    st.subheader("💊 药物排序结果")
    
    res = st.session_state.get("final_drug_rank", pd.DataFrame())
    
    if not res.empty:
        # 聚合评分逻辑
        # 1. 计数: 多少个 GSE 支持
        # 2. 平均分: (注意 L1000和Enrichr分数尺度不同，这里仅作演示，实际需归一化)
        
        # 简单的清洗药名
        res["drug_clean"] = res["drug"].str.lower().str.split("-").str[0]
        
        agg_df = res.groupby("drug_clean").agg(
            Count=('gse', 'nunique'),
            Sources=('source', lambda x: set(x)),
            GSEs=('gse', lambda x: ", ".join(set(x)))
        ).reset_index()
        
        agg_df = agg_df.sort_values("Count", ascending=False)
        
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            st.dataframe(agg_df, use_container_width=True, height=600)
            
        with col_res2:
            st.markdown("### 📊 统计概览")
            st.metric("发现药物总数", len(agg_df))
            st.metric("高频药物 (>1 GSE)", len(agg_df[agg_df["Count"] > 1]))
            
            # 下载按钮
            csv = agg_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 下载完整药物列表 (CSV)",
                csv,
                "drug_repurposing_results.csv",
                "text/csv",
                key='download-csv'
            )
            
            st.markdown("---")
            st.markdown("**下一步建议:**")
            st.markdown("1. 下载 CSV 文件")
            st.markdown("2. 将药物列表导入 PubChem 批量查询结构")
            st.markdown("3. 进行分子对接 (Docking) 验证")
            
    else:
        st.info("请先在 Tab 2 运行分析流程。")

# 清理临时文件 (可选)
# shutil.rmtree(WORK_DIR, ignore_errors=True)
