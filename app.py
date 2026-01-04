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
st.set_page_config(page_title="GEO Drug Repurposing Pipeline", layout="wide", page_icon="💊")

# 定义工作目录
WORK_DIR = Path("workspace")
RAW_DIR = WORK_DIR / "raw"
PROC_DIR = WORK_DIR / "proc"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Session State 初始化
if "geo_hits" not in st.session_state:
    st.session_state["geo_hits"] = pd.DataFrame()
if "selected_gses" not in st.session_state:
    st.session_state["selected_gses"] = []
if "final_drug_rank" not in st.session_state:
    st.session_state["final_drug_rank"] = pd.DataFrame()

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
        # 去除Ensembl版本号 或 /// 分隔符
        g = g.split(".")[0].split("//")[0].strip().upper()
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out

# --- GEO 下载与解析 ---

def geo_search(query, retmax=30):
    """搜索 GEO 数据集"""
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
        st.error(f"Search failed: {e}")
        return pd.DataFrame()

def download_file(url, path):
    """下载文件，如果存在则跳过"""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0: return
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)
    except Exception as e:
        if path.exists(): path.unlink() # 删除损坏文件
        raise e

def get_geo_urls(gse):
    """生成下载链接"""
    gse = gse.strip().upper()
    # 提取数字部分用于构建目录，例如 GSE12345 -> GSE12nnn
    num = re.findall(r'\d+', gse)
    if not num: return "", ""
    series_id = int(num[0])
    prefix = f"GSE{series_id // 1000}nnn"
    
    soft_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/soft/{gse}_family.soft.gz"
    matrix_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/matrix/{gse}_series_matrix.txt.gz"
    return soft_url, matrix_url

def parse_soft_robust(soft_path, case_terms, ctrl_terms):
    """
    增强版 Soft 解析：读取 Title, Source, Characteristics, Description
    返回: (conditions_series, debug_info_dict)
    """
    meta = {}
    current_gsm = None
    
    # 逐行读取 Soft 文件
    with gzip.open(soft_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith("^SAMPLE ="):
                current_gsm = line.split("=")[1].strip()
                meta[current_gsm] = [] # 使用列表存储该样本的所有描述文本
            elif current_gsm:
                # 抓取所有可能包含分组信息的字段
                if line.startswith(("!Sample_title", "!Sample_source_name", "!Sample_characteristics", "!Sample_description")):
                    try:
                        content = line.split("=", 1)[1].strip().lower()
                        meta[current_gsm].append(content)
                    except:
                        pass

    conditions = {}
    debug_info = {} # 用于在界面上展示，帮助用户Debug

    for gsm, texts in meta.items():
        full_text = " | ".join(texts) # 合并所有信息
        debug_info[gsm] = full_text   # 保存给用户看
        
        # 匹配逻辑
        hit_case = any(t in full_text for t in case_terms)
        hit_ctrl = any(t in full_text for t in ctrl_terms)
        
        if hit_case and not hit_ctrl:
            conditions[gsm] = "case"
        elif hit_ctrl and not hit_case:
            conditions[gsm] = "control"
        elif hit_case and hit_ctrl:
            # 冲突处理：通常 Case 的描述（如 specific mutation）比 Control 更特异
            # 如果包含 disease/mutation，由于 control 样本也可能提到 disease (e.g. "control for disease X")
            # 这里保守起见设为 ambiguous，或者你可以偏向 Case
            conditions[gsm] = "ambiguous"
        else:
            conditions[gsm] = "unknown"
            
    return pd.Series(conditions), debug_info

# --- 差异分析主流程 ---

def run_analysis_pipeline(gse, case_terms, ctrl_terms):
    """下载 -> 解析 -> 差异分析"""
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
        return None, f"Download Error: {str(e)}", {}

    # 2. 分组解析 (关键步骤)
    conditions, debug_info = parse_soft_robust(soft_path, case_terms, ctrl_terms)
    
    case_samps = conditions[conditions == "case"].index.tolist()
    ctrl_samps = conditions[conditions == "control"].index.tolist()
    
    # 如果分组失败，直接返回调试信息
    if len(case_samps) == 0 or len(ctrl_samps) == 0:
        msg = f"Insufficient Samples: Case={len(case_samps)}, Ctrl={len(ctrl_samps)}"
        return None, msg, debug_info
    
    # 3. 读取矩阵
    try:
        # matrix文件通常 header比较乱，skiprows=... 需要自动判断，这里假设标准格式 !series_matrix_table_begin 下一行是header
        # 简单处理：直接 read_csv, comment='!'
        df = pd.read_csv(matrix_path, sep="\t", comment="!", index_col=0, on_bad_lines='skip')
        df = df.dropna(how='all')
        df = df.fillna(0)
        
        # 简单的数据变换判断
        if df.max().max() > 50:
            df = np.log2(df + 1)
    except Exception as e:
        return None, f"Matrix Parse Error: {str(e)}", debug_info
    
    # 对齐
    # 矩阵列名可能是 GSMxxxxx 也可能是 "GSMxxxxx_sample_name"，做模糊匹配
    valid_cols = []
    col_map = {} # Matrix Col -> GSM
    
    for col in df.columns:
        # 尝试提取 col 中的 GSM
        m = re.search(r'(GSM\d+)', col)
        if m:
            gsm = m.group(1)
            if gsm in conditions.index:
                valid_cols.append(col)
                col_map[col] = gsm
    
    if len(valid_cols) < 2:
        return None, f"Column Mismatch: Matrix columns do not match SOFT GSM IDs. Found: {list(df.columns[:5])}", debug_info
    
    df = df[valid_cols]
    
    # 映射回 condition
    case_cols = [c for c in valid_cols if conditions.get(col_map[c]) == "case"]
    ctrl_cols = [c for c in valid_cols if conditions.get(col_map[c]) == "control"]
    
    if len(case_cols) == 0 or len(ctrl_cols) == 0:
         return None, f"Aligned Samples Missing: Case={len(case_cols)}, Ctrl={len(ctrl_cols)}", debug_info

    # 4. 差异分析 (T-test 或 Mean Diff)
    results = []
    use_ttest = len(case_cols) >= 2 and len(ctrl_cols) >= 2
    
    # 为速度考虑，如果不使用 pydeseq2，这里用 numpy 向量化计算会更快
    # 这里用 iterrows 虽然慢点但稳健
    for gene, row in df.iterrows():
        case_vals = row[case_cols].values
        ctrl_vals = row[ctrl_cols].values
        
        diff = np.mean(case_vals) - np.mean(ctrl_vals)
        p = 1.0
        
        if use_ttest:
            # 忽略全为0或方差极小的情况
            if np.std(case_vals) < 1e-6 and np.std(ctrl_vals) < 1e-6:
                p = 1.0
            else:
                try:
                    _, p = stats.ttest_ind(case_vals, ctrl_vals, equal_var=False)
                except:
                    p = 1.0
        
        results.append({"gene": gene, "log2fc": diff, "pval": p})
        
    res_df = pd.DataFrame(results)
    if res_df.empty: return None, "No valid DE results", debug_info
    
    # FDR 校正
    res_df["pval"] = res_df["pval"].fillna(1.0)
    res_df["padj"] = multipletests(res_df["pval"], method="fdr_bh")[1]
    res_df = res_df.sort_values("log2fc", key=abs, ascending=False) # 按 LogFC 绝对值排序
    
    # 提取基因名 (去除 /// 或 ID)
    res_df["gene_symbol"] = res_df["gene"].apply(lambda x: str(x).split("//")[0].split(".")[0].strip().upper())
    
    return res_df, f"Success: Case={len(case_cols)}, Ctrl={len(ctrl_cols)}", debug_info

# --- Connectivity API ---

def run_l1000fwd(up_genes, dn_genes):
    url = "https://maayanlab.cloud/l1000fwd/sig_search"
    # L1000FWD 对基因数量有限制，且必须是大写 Symbol
    payload = {"up_genes": up_genes[:150], "down_genes": dn_genes[:150]}
    try:
        r = requests.post(url, json=payload, timeout=30)
        res_id = r.json().get("result_id")
        if not res_id: return pd.DataFrame()
        
        time.sleep(1)
        r2 = requests.get(f"https://maayanlab.cloud/l1000fwd/result/topn/{res_id}", timeout=30)
        data = r2.json()
        
        rows = []
        # 我们主要关注 'opposite' (反转 gene signature 的药物)
        if "opposite" in data:
            for item in data["opposite"]:
                rows.append({
                    "drug": item.get("pert_id"), # L1000FWD 返回的是 ID 或 Name
                    "score": item.get("score"),
                    "source": "L1000FWD",
                    "direction": "opposite"
                })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"L1000FWD Error: {e}")
        return pd.DataFrame()

def run_enrichr(genes, library="LINCS_L1000_Chem_Pert_down"):
    base = "https://maayanlab.cloud/Enrichr"
    try:
        # 1. Add List
        r = requests.post(f"{base}/addList", files={
            'list': (None, '\n'.join(genes[:300])),
            'description': (None, 'Streamlit_Pipeline')
        }, timeout=30)
        user_list_id = r.json().get("userListId")
        if not user_list_id: return pd.DataFrame()
        
        # 2. Enrich
        r2 = requests.get(f"{base}/enrich?userListId={user_list_id}&backgroundType={library}", timeout=30)
        data = r2.json()
        if library not in data: return pd.DataFrame()
        
        rows = []
        for item in data[library]:
            # Enrichr 结果格式: [Rank, Term, P-value, Z-score, Combined Score, ...]
            # Term 通常是 "DrugName_CellLine_..."
            term = item[1]
            drug_name = term.split("_")[0].split(" ")[0] # 简单清洗
            
            rows.append({
                "drug": drug_name,
                "score": item[4], # Combined Score
                "pval": item[2],
                "source": "Enrichr"
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Enrichr Error: {e}")
        return pd.DataFrame()

# ==========================================
# 2. Streamlit 界面逻辑
# ==========================================

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    taxon_filter = st.selectbox("物种过滤", ["Homo sapiens", "Mus musculus", "All"], index=0)
    
    st.divider()
    st.markdown("### 🏷️ 分组关键词 (关键)")
    st.info("如果找不到样本，请在这里添加样本描述中出现的词。")
    
    # 针对你之前 CLCN / Cystic Fibrosis 优化的默认关键词
    default_case = "mutation, mutant, variant, patient, knockout, knockdown, disease, clcn, cf, cystic fibrosis, tumor, cancer, treated, stimulation, infected"
    default_ctrl = "control, wt, wild type, wild-type, healthy, normal, vehicle, pbs, dmso, mock, baseline, untreated, placebo, non-targeting"
    
    case_input = st.text_area("实验组 (Case) 关键词", default_case, height=100)
    ctrl_input = st.text_area("对照组 (Control) 关键词", default_ctrl, height=100)
    
    case_terms = [x.strip().lower() for x in case_input.split(",") if x.strip()]
    ctrl_terms = [x.strip().lower() for x in ctrl_input.split(",") if x.strip()]
    
    st.divider()
    top_n_genes = st.number_input("Signature 基因数量 (Top N)", 50, 500, 150)
    st.caption("提取多少个差异基因用于药物预测")

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["1️⃣ 搜索 & 选择", "2️⃣ 运行批处理", "3️⃣ 结果看板"])

# --- Tab 1: Search ---
with tab1:
    st.subheader("🔍 搜索 GEO 数据集")
    col1, col2 = st.columns([3, 1])
    with col1:
        query_text = st.text_input("输入查询", value='(CLCN2 OR "chloride channel 2") AND (mutation OR knockout) AND "RNA-seq"')
    with col2:
        search_btn = st.button("开始搜索", use_container_width=True)
    
    if search_btn and query_text:
        with st.spinner("正在连接 NCBI..."):
            df_hits = geo_search(query_text)
            if not df_hits.empty:
                if taxon_filter != "All":
                    df_hits = df_hits[df_hits["Taxon"] == taxon_filter]
                st.session_state["geo_hits"] = df_hits
            else:
                st.warning("未找到结果，请放宽关键词或检查网络。")
    
    if not st.session_state["geo_hits"].empty:
        st.write(f"找到 {len(st.session_state['geo_hits'])} 个数据集 (请勾选要分析的):")
        
        # 使用 DataEditor 选择
        hits_display = st.session_state["geo_hits"].copy()
        hits_display.insert(0, "Select", False)
        
        edited_df = st.data_editor(
            hits_display,
            column_config={"Select": st.column_config.CheckboxColumn(required=True)},
            disabled=["Accession", "Title", "Summary", "Taxon", "Samples", "Date"],
            use_container_width=True,
            hide_index=True
        )
        
        selected = edited_df[edited_df["Select"]]["Accession"].tolist()
        st.session_state["selected_gses"] = selected
        
        if selected:
            st.success(f"已选择 {len(selected)} 个数据集: {', '.join(selected)}")
            st.info("👉 请前往 '2️⃣ 运行批处理' 标签页开始分析")
    else:
        st.write("暂无数据。")

# --- Tab 2: Run ---
with tab2:
    st.subheader("⚡ 自动化分析 Pipeline")
    
    if not st.session_state["selected_gses"]:
        st.warning("⚠️ 请先在第 1 步选择至少一个数据集。")
    else:
        st.markdown(f"**待分析列表**: {', '.join(st.session_state['selected_gses'])}")
        
        if st.button("🚀 启动分析 (Start Pipeline)", type="primary"):
            results_bucket = []
            log_area = st.container()
            progress_bar = st.progress(0)
            
            total = len(st.session_state["selected_gses"])
            
            for i, gse in enumerate(st.session_state["selected_gses"]):
                with log_area:
                    st.write(f"--- 处理中: **{gse}** ({i+1}/{total}) ---")
                    
                    # 1. 差异分析
                    df_de, msg, debug_info = run_analysis_pipeline(gse, case_terms, ctrl_terms)
                    
                    if df_de is None:
                        st.error(f"❌ {gse} 失败: {msg}")
                        # === DEBUG 关键点 ===
                        with st.expander(f"🕵️‍♂️ 调试: {gse} 的样本元数据 (为什么没匹配到?)"):
                            st.caption("系统读取到的样本描述如下。请检查这些文本，找出代表 Case/Control 的特定词汇，并添加到左侧设置栏。")
                            # 只显示前 15 个样本，避免太长
                            preview_keys = list(debug_info.keys())[:15]
                            st.json({k: debug_info[k] for k in preview_keys})
                        continue
                    
                    st.success(f"✅ {gse} 差异分析完成: {msg}")
                    
                    # 2. 提取 Signature
                    # 只有 LogFC 大的才算 Up，小的才算 Down
                    up_genes = df_de[df_de["log2fc"] > 0].head(top_n_genes)["gene_symbol"].tolist()
                    dn_genes = df_de[df_de["log2fc"] < 0].tail(top_n_genes)["gene_symbol"].tolist()
                    
                    up_genes = clean_gene_list(up_genes)
                    dn_genes = clean_gene_list(dn_genes)
                    
                    if len(up_genes) < 10 or len(dn_genes) < 10:
                        st.warning(f"⚠️ {gse}: 差异基因过少 (Up={len(up_genes)}, Down={len(dn_genes)})，跳过药物预测。")
                        continue
                        
                    # 3. 药物预测 API
                    st.text(f"正在查询 L1000FWD 和 Enrichr...")
                    
                    # L1000FWD (找反转)
                    df_l1000 = run_l1000fwd(up_genes, dn_genes)
                    
                    # Enrichr (UP genes vs Drug Down) -> Reversal
                    df_enrichr = run_enrichr(up_genes, library="LINCS_L1000_Chem_Pert_down")
                    # Enrichr (Down genes vs Drug Up) -> Reversal (Optional, add if needed)
                    
                    # 合并
                    parts = []
                    if not df_l1000.empty: parts.append(df_l1000)
                    if not df_enrichr.empty: parts.append(df_enrichr)
                    
                    if parts:
                        combined = pd.concat(parts)
                        combined["gse"] = gse
                        results_bucket.append(combined)
                        with st.expander(f"💊 {gse} 预测到的 Top 药物"):
                            st.dataframe(combined.head(5))
                    else:
                        st.warning(f"{gse}: API 未返回有效药物结果。")
                
                progress_bar.progress((i + 1) / total)
            
            st.success("🎉 所有任务处理完毕！请查看 '3️⃣ 结果看板'")
            if results_bucket:
                st.session_state["final_drug_rank"] = pd.concat(results_bucket)
            else:
                st.session_state["final_drug_rank"] = pd.DataFrame()

# --- Tab 3: Results ---
with tab3:
    st.subheader("💊 药物汇总与排序")
    
    res = st.session_state.get("final_drug_rank", pd.DataFrame())
    
    if not res.empty:
        # 清洗药名 (转小写，去除非法字符)
        res["drug_clean"] = res["drug"].astype(str).str.lower().str.strip()
        # 去掉 BRD-xxxx 这种内部ID，如果太短的通常不是好药名
        res = res[res["drug_clean"].str.len() > 2]
        
        # 聚合统计
        agg_df = res.groupby("drug_clean").agg(
            Frequency=('gse', 'nunique'),         # 在多少个 GSE 中出现
            Total_Score=('score', 'sum'),         # 总分 (仅供参考，不同源分数不可直接加)
            Sources=('source', lambda x: ", ".join(sorted(set(x)))),
            Support_GSEs=('gse', lambda x: ", ".join(sorted(set(x))))
        ).reset_index()
        
        # 排序：优先按出现频率，其次按总分
        agg_df = agg_df.sort_values(["Frequency", "Total_Score"], ascending=[False, False])
        agg_df.columns = ["Drug Name", "GSE Count", "Sum Score", "Sources", "GSE IDs"]
        
        # 展示 Top 结果
        col_view, col_stat = st.columns([3, 1])
        
        with col_view:
            st.markdown("### 🏆 Top 候选药物列表")
            st.dataframe(
                agg_df.style.background_gradient(subset=["GSE Count"], cmap="Greens"),
                use_container_width=True,
                height=600
            )
        
        with col_stat:
            st.markdown("### 📊 统计")
            st.metric("总药物数", len(agg_df))
            st.metric("高置信度 (>1 GSE)", len(agg_df[agg_df["GSE Count"] > 1]))
            
            st.download_button(
                "📥 下载完整 CSV",
                data=agg_df.to_csv(index=False).encode("utf-8"),
                file_name="drug_repurposing_final_rank.csv",
                mime="text/csv",
                type="primary"
            )
            
            st.markdown("---")
            st.info("提示: GSE Count 越高，代表该药物在多个独立数据集中均显示出对疾病特征的反转作用，可靠性越高。")
            
    else:
        st.info("暂无结果。请先运行 Pipeline，并确保至少有一个数据集成功跑通。")
