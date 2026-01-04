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
        if path.exists(): path.unlink()
        raise e

def get_geo_urls(gse):
    """生成下载链接"""
    gse = gse.strip().upper()
    num = re.findall(r'\d+', gse)
    if not num: return "", ""
    series_id = int(num[0])
    prefix = f"GSE{series_id // 1000}nnn"
    
    soft_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/soft/{gse}_family.soft.gz"
    matrix_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/matrix/{gse}_series_matrix.txt.gz"
    return soft_url, matrix_url

def extract_metadata_only(gse):
    """
    只下载并解析 SOFT 文件，用于预览
    返回: DataFrame (GSM, Title, Full_Text)
    """
    gse_dir = RAW_DIR / gse
    gse_dir.mkdir(exist_ok=True)
    soft_url, _ = get_geo_urls(gse)
    soft_path = gse_dir / f"{gse}_family.soft.gz"
    
    # 下载
    try:
        download_file(soft_url, soft_path)
    except Exception as e:
        return None, str(e)

    # 解析
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
        
        # Add last one
        if current_gsm:
             meta.append({
                "GSM": current_data["GSM"],
                "Title": current_data["Title"],
                "Full_Description": " | ".join(current_data["Text"]).lower()
            })
            
    return pd.DataFrame(meta), "Success"

# def determine_group(text, case_terms, ctrl_terms):
#     text = text.lower()
#     hit_case = any(t in text for t in case_terms)
#     hit_ctrl = any(t in text for t in ctrl_terms)
    
#     if hit_case and not hit_ctrl: return "Case", "red"
#     if hit_ctrl and not hit_case: return "Control", "green"
#     if hit_case and hit_ctrl: return "Ambiguous (Both)", "orange"
#     return "Unknown", "grey"



def determine_group(text, case_terms, ctrl_terms):
    text = text.lower()
    hit_case = any(t in text for t in case_terms)
    hit_ctrl = any(t in text for t in ctrl_terms)
    
    # === 修改开始：冲突处理逻辑 ===
    if hit_case and hit_ctrl:
        # 这里的逻辑是：如果一个样本既说自己是 mutation 又说自己是 control
        # 通常它是对照组（例如 "Control sample from mutation patient"）
        # 所以我们优先判定为 Control
        return "Control", "#e6ffe6"  # 浅绿色背景
    # === 修改结束 ===

    if hit_case and not hit_ctrl: return "Case", "#ffe6e6" # 浅红色
    if hit_ctrl and not hit_case: return "Control", "#e6ffe6" # 浅绿色
    
    return "Unknown", "grey"



# --- 差异分析主流程 ---

def run_analysis_pipeline(gse, case_terms, ctrl_terms):
    gse_dir = RAW_DIR / gse
    gse_dir.mkdir(exist_ok=True)
    
    soft_url, matrix_url = get_geo_urls(gse)
    soft_path = gse_dir / f"{gse}_family.soft.gz"
    matrix_path = gse_dir / f"{gse}_series_matrix.txt.gz"
    
    # 1. 下载 (确保文件都存在)
    try:
        download_file(soft_url, soft_path)
        download_file(matrix_url, matrix_path)
    except Exception as e:
        return None, f"Download Error: {str(e)}"

    # 2. 复用 metadata 解析逻辑
    df_meta, msg = extract_metadata_only(gse)
    if df_meta is None or df_meta.empty:
        return None, f"Metadata Parse Error: {msg}"
    
    # 3. 确定分组
    conditions = {}
    for idx, row in df_meta.iterrows():
        group, _ = determine_group(row["Full_Description"], case_terms, ctrl_terms)
        if group == "Case": conditions[row["GSM"]] = "case"
        elif group == "Control": conditions[row["GSM"]] = "control"
    
    case_samps = [k for k,v in conditions.items() if v == "case"]
    ctrl_samps = [k for k,v in conditions.items() if v == "control"]
    
    if len(case_samps) == 0 or len(ctrl_samps) == 0:
        return None, f"Insufficient Samples: Case={len(case_samps)}, Ctrl={len(ctrl_samps)}"
    
    # 4. 读取矩阵
    try:
        # 更加鲁棒的读取方式
        df = pd.read_csv(matrix_path, sep="\t", comment="!", index_col=0, on_bad_lines='skip')
        
        # 移除全是 NaN 的行/列
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        
        # 强制转 numeric，无法转换的变为 NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(how='any') # 只要有一个样本是NaN，这个基因就扔掉，防止报错
        
        # 简单的数据变换判断
        if not df.empty and df.max().max() > 50:
            df = np.log2(df + 1)
            
    except Exception as e:
        return None, f"Matrix Parse Error: {str(e)}"
    
    # 对齐
    col_map = {} 
    valid_cols = []
    
    # 尝试匹配列名 (列名可能是 GSM12345 或 "Sample 1 (GSM12345)")
    for col in df.columns:
        m = re.search(r'(GSM\d+)', str(col))
        if m:
            gsm = m.group(1)
            if gsm in conditions:
                col_map[col] = gsm
                valid_cols.append(col)
    
    if len(valid_cols) < 2:
        return None, f"Column Mismatch. Matrix cols: {list(df.columns[:3])}... vs Soft IDs: {list(conditions.keys())[:3]}..."
    
    df = df[valid_cols]
    
    case_cols = [c for c in valid_cols if conditions[col_map[c]] == "case"]
    ctrl_cols = [c for c in valid_cols if conditions[col_map[c]] == "control"]
    
    if len(case_cols) < 1 or len(ctrl_cols) < 1:
         return None, f"Aligned Samples Missing: Case={len(case_cols)}, Ctrl={len(ctrl_cols)}"

    # 5. 差异分析
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
    if res_df.empty: return None, "No valid DE results (dataframe empty)"
    
    res_df["pval"] = res_df["pval"].fillna(1.0)
    res_df["padj"] = multipletests(res_df["pval"], method="fdr_bh")[1]
    res_df = res_df.sort_values("log2fc", key=abs, ascending=False)
    
    # 提取基因名
    res_df["gene_symbol"] = res_df["gene"].apply(lambda x: str(x).split("//")[0].split(".")[0].strip().upper())
    
    return res_df, f"Success: Case={len(case_cols)}, Ctrl={len(ctrl_cols)}"

# --- Connectivity API ---

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
                rows.append({
                    "drug": item.get("pert_id"),
                    "score": item.get("score"),
                    "source": "L1000FWD",
                    "direction": "opposite"
                })
        return pd.DataFrame(rows)
    except: return pd.DataFrame()

def run_enrichr(genes, library="LINCS_L1000_Chem_Pert_down"):
    base = "https://maayanlab.cloud/Enrichr"
    try:
        r = requests.post(f"{base}/addList", files={'list': (None, '\n'.join(genes[:300])), 'description': (None, 'Streamlit')}, timeout=30)
        user_list_id = r.json().get("userListId")
        if not user_list_id: return pd.DataFrame()
        r2 = requests.get(f"{base}/enrich?userListId={user_list_id}&backgroundType={library}", timeout=30)
        data = r2.json()
        if library not in data: return pd.DataFrame()
        rows = []
        for item in data[library]:
            rows.append({
                "drug": item[1].split("_")[0].split(" ")[0],
                "score": item[4],
                "pval": item[2],
                "source": "Enrichr"
            })
        return pd.DataFrame(rows)
    except: return pd.DataFrame()

# ==========================================
# 2. Streamlit 界面逻辑
# ==========================================

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ 全局分组设置")
    st.info("💡 提示：请先在右侧 '🔬 样本调试器' 中找到数据集中使用的特定词汇，然后复制到这里。")
    
    default_case = "mutation, mutant, variant, patient, knockout, knockdown, disease, clcn, cf, cystic fibrosis, tumor, cancer, treated, stimulation, infected"
    default_ctrl = "control, wt, wild type, wild-type, healthy, normal, vehicle, pbs, dmso, mock, baseline, untreated, placebo, non-targeting"
    
    case_input = st.text_area("Case (实验组) 关键词", default_case, height=120)
    ctrl_input = st.text_area("Control (对照组) 关键词", default_ctrl, height=120)
    
    case_terms = [x.strip().lower() for x in case_input.split(",") if x.strip()]
    ctrl_terms = [x.strip().lower() for x in ctrl_input.split(",") if x.strip()]
    
    st.divider()
    top_n_genes = st.number_input("Signature Top N", 50, 500, 150)
    taxon_filter = st.selectbox("物种", ["Homo sapiens", "Mus musculus", "All"], index=0)

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ 搜索数据集", "2️⃣ 🔬 样本分组调试器", "3️⃣ ⚡ 运行分析", "4️⃣ 📊 结果看板"])

# --- Tab 1: Search ---
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        query_text = st.text_input("GEO Search Query", value='(CLCN2 OR "chloride channel 2") AND (mutation OR knockout) AND "RNA-seq"')
    with col2:
        if st.button("开始搜索", use_container_width=True):
            with st.spinner("Searching NCBI..."):
                df = geo_search(query_text)
                if not df.empty and taxon_filter != "All":
                    df = df[df["Taxon"] == taxon_filter]
                st.session_state["geo_hits"] = df

    if not st.session_state["geo_hits"].empty:
        st.write(f"Found {len(st.session_state['geo_hits'])} datasets:")
        # Data Editor
        hits = st.session_state["geo_hits"].copy()
        hits.insert(0, "Select", False)
        edited = st.data_editor(hits, column_config={"Select": st.column_config.CheckboxColumn(required=True)}, disabled=["Accession", "Title"], use_container_width=True, hide_index=True)
        st.session_state["selected_gses"] = edited[edited["Select"]]["Accession"].tolist()
        if st.session_state["selected_gses"]:
            st.success(f"已选择: {st.session_state['selected_gses']} (请前往 Tab 2 预览分组)")

# --- Tab 2: Metadata Inspector (New!) ---
with tab2:
    st.subheader("🔬 样本元数据深度预览 & 关键词提取")
    st.markdown("在这里检查你的关键词是否能正确匹配样本。**先解决 'Unknown' 和 'Case=0/Ctrl=0' 的问题，再运行 Pipeline。**")
    
    if not st.session_state["selected_gses"]:
        st.warning("请先在 Tab 1 选择数据集。")
    else:
        # Selector
        inspect_gse = st.selectbox("选择要调试的数据集:", st.session_state["selected_gses"])
        
        if st.button(f"🔍 获取 {inspect_gse} 的元数据"):
            with st.spinner("正在下载描述文件 (不下载矩阵)..."):
                df_meta, msg = extract_metadata_only(inspect_gse)
                if df_meta is not None:
                    st.session_state["metadata_cache"][inspect_gse] = df_meta
                    st.success("元数据加载成功！")
                else:
                    st.error(f"加载失败: {msg}")
        
        # Display Logic
        if inspect_gse in st.session_state["metadata_cache"]:
            df_display = st.session_state["metadata_cache"][inspect_gse].copy()
            
            # 实时计算分组状态
            df_display["Predicted Group"] = df_display["Full_Description"].apply(
                lambda x: determine_group(x, case_terms, ctrl_terms)[0]
            )
            
            # 统计
            counts = df_display["Predicted Group"].value_counts()
            c_case = counts.get("Case", 0)
            c_ctrl = counts.get("Control", 0)
            c_unk = counts.get("Unknown", 0)
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Samples", len(df_display))
            m1.metric("Dataset", inspect_gse)
            m2.metric("✅ Matched Case", c_case)
            m3.metric("✅ Matched Control", c_ctrl)
            m4.metric("❌ Unknown", c_unk, delta_color="inverse")
            
            if c_case == 0 or c_ctrl == 0:
                st.error("⚠️ 警告: 缺少实验组或对照组！请从下表中寻找关键词并添加到左侧侧边栏。")
            else:
                st.success("状态良好，可以进行分析。")
            
            # 自动提取关键词建议
            all_text = " ".join(df_display["Full_Description"].tolist()).lower()
            # 简单分词，去掉常用词
            words = re.findall(r'\b[a-z]{3,}\b', all_text)
            common_stops = set(["the","and","for","with","from","sample","rna","seq","homo","sapiens","mus","musculus","extraction","total","analysis","description","characteristics","source","name","title","geo","accession","platform","organism","instrument","model","library","strategy","layout"])
            filtered_words = [w for w in words if w not in common_stops and w not in case_terms and w not in ctrl_terms]
            most_common = Counter(filtered_words).most_common(20)
            
            with st.expander("💡 关键词推荐 (点击复制到侧边栏)"):
                st.write("以下是元数据中出现频率最高、且不在你当前列表中的词汇。如果看到具体的疾病名或药物名，请手动添加到左侧。")
                st.code(", ".join([f"{w[0]}" for w in most_common]), language="text")

            # 主表格
            def color_row(row):
                grp = row["Predicted Group"]
                if grp == "Case": return ['background-color: #ffe6e6'] * len(row)
                if grp == "Control": return ['background-color: #e6ffe6'] * len(row)
                if grp == "Unknown": return ['background-color: #f0f0f0'] * len(row)
                return [''] * len(row)

            st.dataframe(
                df_display.style.apply(color_row, axis=1),
                column_config={
                    "Full_Description": st.column_config.TextColumn("Sample Description", width="large"),
                    "Predicted Group": st.column_config.TextColumn("Current Status", width="medium")
                },
                use_container_width=True,
                height=500
            )

# --- Tab 3: Run ---
with tab3:
    st.subheader("⚡ 批处理分析")
    
    if st.button("🚀 启动分析 (Start Pipeline)", type="primary"):
        results_bucket = []
        log_container = st.container()
        progress = st.progress(0)
        
        for i, gse in enumerate(st.session_state["selected_gses"]):
            with log_container:
                st.write(f"**Processing {gse} ({i+1}/{len(st.session_state['selected_gses'])})**...")
                
                df_de, msg = run_analysis_pipeline(gse, case_terms, ctrl_terms)
                
                if df_de is None:
                    st.error(f"❌ {gse} Failed: {msg}")
                    continue
                
                st.success(f"✅ {gse} DE Done. Genes: {len(df_de)}")
                
                # Signature
                up = df_de[df_de["log2fc"] > 0].head(top_n_genes)["gene_symbol"].tolist()
                dn = df_de[df_de["log2fc"] < 0].tail(top_n_genes)["gene_symbol"].tolist()
                
                if len(up) < 5 or len(dn) < 5:
                    st.warning(f"Not enough DE genes for {gse}")
                    continue
                
                # API
                df_l = run_l1000fwd(clean_gene_list(up), clean_gene_list(dn))
                df_e = run_enrichr(clean_gene_list(up), "LINCS_L1000_Chem_Pert_down")
                
                comb = pd.concat([df_l, df_e])
                if not comb.empty:
                    comb["gse"] = gse
                    results_bucket.append(comb)
            
            progress.progress((i+1)/len(st.session_state["selected_gses"]))
        
        if results_bucket:
            st.session_state["final_drug_rank"] = pd.concat(results_bucket)
            st.success("Pipeline Finished! Check Tab 4.")
        else:
            st.warning("No drugs found.")

# --- Tab 4: Results ---
with tab4:
    res = st.session_state.get("final_drug_rank", pd.DataFrame())
    if not res.empty:
        res["drug_clean"] = res["drug"].astype(str).str.lower().str.strip()
        agg = res.groupby("drug_clean").agg(
            Count=('gse', 'nunique'),
            Score_Sum=('score', 'sum'),
            GSEs=('gse', lambda x: ",".join(set(x))),
            Sources=('source', lambda x: ",".join(set(x)))
        ).reset_index().sort_values(["Count", "Score_Sum"], ascending=[False, False])
        
        st.dataframe(agg, use_container_width=True)
        st.download_button("Download CSV", agg.to_csv().encode("utf-8"), "drugs.csv")
    else:
        st.info("No results yet.")
