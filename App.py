"""
SOYSIGNAL v1.0 - NLP Sentiment Dashboard
Soybean commodity news sentiment analysis & market price trends (Jan 2014 - Dec 2020)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

# ─── Scikit-learn imports ────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SOYSIGNAL – NLP Sentiment",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette ─────────────────────────────────────────────────────────────
C_BG       = "#0d1117"
C_CARD     = "#161b22"
C_BORDER   = "#21262d"
C_ACCENT   = "#00e5ff"
C_GREEN    = "#39d353"
C_RED      = "#f85149"
C_YELLOW   = "#e3b341"
C_TEXT     = "#c9d1d9"
C_MUTED    = "#8b949e"

LABEL_COLORS = {-1: C_RED, 0: C_YELLOW, 1: C_GREEN}
LABEL_NAMES  = {-1: "Down (-1)", 0: "Neutral (0)", 1: "Up (+1)"}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Global ── */
  html, body, [class*="css"] {{
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    background-color: {C_BG};
    color: {C_TEXT};
  }}
  .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background-color: #0d1117;
    border-right: 1px solid {C_BORDER};
  }}
  [data-testid="stSidebar"] .stRadio label {{
    color: {C_MUTED};
    font-size: 0.82rem;
    padding: 4px 0;
    cursor: pointer;
  }}
  [data-testid="stSidebar"] .stRadio label:hover {{ color: {C_ACCENT}; }}

  /* ── Metric cards ── */
  .metric-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 18px 20px;
    text-align: center;
  }}
  .metric-label {{ font-size: 0.68rem; color: {C_MUTED}; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; }}
  .metric-value {{ font-size: 2.2rem; font-weight: 700; line-height: 1.1; }}
  .metric-sub   {{ font-size: 0.72rem; color: {C_MUTED}; margin-top: 4px; }}

  /* ── Section headers ── */
  .section-header {{
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: {C_MUTED};
    text-transform: uppercase;
    border-bottom: 1px solid {C_BORDER};
    padding-bottom: 6px;
    margin: 24px 0 14px 0;
  }}

  /* ── Pill badge ── */
  .pill {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
  }}
  .pill-red    {{ background:#3d1c1c; color:{C_RED};    border:1px solid {C_RED};    }}
  .pill-green  {{ background:#1c3d22; color:{C_GREEN};  border:1px solid {C_GREEN};  }}
  .pill-yellow {{ background:#3d3216; color:{C_YELLOW}; border:1px solid {C_YELLOW}; }}
  .pill-blue   {{ background:#1c2d3d; color:{C_ACCENT}; border:1px solid {C_ACCENT}; }}

  /* ── Pipeline nodes ── */
  .pipe-row {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:14px 0; }}
  .pipe-node {{
    border:1px solid {C_ACCENT};
    color:{C_ACCENT};
    border-radius:6px;
    padding:6px 14px;
    font-size:0.78rem;
    background: rgba(0,229,255,0.06);
  }}
  .pipe-node.active {{
    background: rgba(0,229,255,0.18);
    font-weight:700;
  }}
  .pipe-arrow {{ color:{C_MUTED}; font-size:1rem; }}

  /* ── Data table ── */
  .stDataFrame {{ border-radius:8px; overflow:hidden; }}

  /* ── Divider ── */
  .custom-hr {{ border:none; border-top:1px solid {C_BORDER}; margin:20px 0; }}

  /* ── Prediction result ── */
  .pred-box {{
    border-radius:10px;
    padding:24px;
    text-align:center;
    margin-top:16px;
  }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df["monthly_average"] = df["monthly_average"].str.replace(",", ".").astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    hd_cols = [f"hd{i}" for i in range(1, 11)]
    df["headline_count"] = df[hd_cols].notna().sum(axis=1)

    # headline summary (first ~60 chars of raw_text)
    df["headline_summary"] = df["raw_text"].str[:65] + "…"
    return df

@st.cache_resource
def train_model(df):
    X, y = df["clean_text"], df["Label3"]

    # TF-IDF with bigrams
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500, min_df=2)
    X_vec = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=7, stratify=y
    )

    dt = DecisionTreeClassifier(max_depth=8, min_samples_split=4, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
    cr  = classification_report(
        y_test, y_pred,
        labels=[-1, 0, 1],
        target_names=["Down(-1)", "Neutral(0)", "Up(+1)"],
        output_dict=True
    )

    # Feature importance
    feat_names = tfidf.get_feature_names_out()
    importance  = dt.feature_importances_
    top_idx     = np.argsort(importance)[::-1][:20]
    top_feats   = [(feat_names[i], importance[i]) for i in top_idx]

    # CV score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(dt, X_vec, y, cv=cv, scoring="accuracy")

    return {
        "model": dt, "vectorizer": tfidf,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred, "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": cr,
        "top_features": top_feats,
        "cv_scores": cv_scores,
    }

df    = load_data()
model = train_model(df)

# ─── matplotlib global style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  C_BG,
    "axes.facecolor":    C_CARD,
    "axes.edgecolor":    C_BORDER,
    "axes.labelcolor":   C_MUTED,
    "xtick.color":       C_MUTED,
    "ytick.color":       C_MUTED,
    "text.color":        C_TEXT,
    "grid.color":        C_BORDER,
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='margin-bottom:6px;'>
      <span style='color:{C_ACCENT}; font-size:1.3rem; font-weight:700;'>○ SOYSIGNAL</span><br/>
      <span style='color:{C_MUTED}; font-size:0.7rem;'>v1.0 · NLP Sentiment</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Overview", "EDA", "Preprocessing", "Feature Extraction",
         "Model Evaluation", "Prediction", "Conclusion"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.65rem; color:{C_MUTED}; line-height:1.8;'>
      SOYBEAN · SENTIMENT · NLP<br/>
      Jan 2014 – Dec 2020<br/>
      Decision Tree + TF-IDF Bigrams
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════
def metric_card(label, value, sub, color=C_ACCENT):
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color};">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>"""

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def pill(text, style="blue"):
    return f'<span class="pill pill-{style}">{text}</span>'

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("## Market Sentiment Dashboard")
    st.markdown(
        f"<span style='color:{C_MUTED}; font-size:0.82rem;'>"
        "Analisis sentimen berita komoditas soybean &amp; tren harga pasar · Jan 2014 – Dec 2020"
        "</span>",
        unsafe_allow_html=True,
    )

    # ── Top metrics ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Total Data", "83", "records"), unsafe_allow_html=True)
    c2.markdown(metric_card("Jumlah Kelas", "3", "down · neutral · up", C_ACCENT), unsafe_allow_html=True)
    c3.markdown(metric_card("Best Model", "DT+N-Grams", "decision tree", C_ACCENT), unsafe_allow_html=True)
    c4.markdown(metric_card("Accuracy", f"{model['accuracy']*100:.2f}%", "test set", C_GREEN), unsafe_allow_html=True)

    # ── Dataset preview ──────────────────────────────────────────────────────
    section("DATASET PREVIEW")
    preview = df[["date", "monthly_average", "Label3", "headline_summary"]].copy()
    preview.columns = ["DATE", "MONTHLY AVG", "LABEL3", "HEADLINE SUMMARY"]
    preview["DATE"] = preview["DATE"].dt.strftime("%Y-%m-%d")
    st.dataframe(preview.head(10), use_container_width=True, hide_index=True)

    # ── Pipeline summary ─────────────────────────────────────────────────────
    section("PIPELINE SUMMARY")
    st.markdown("""
    <div class="pipe-row">
      <span class="pipe-node">Raw Text</span>
      <span class="pipe-arrow">→</span>
      <span class="pipe-node">Cleaning</span>
      <span class="pipe-arrow">→</span>
      <span class="pipe-node">TF-IDF / N-Grams</span>
      <span class="pipe-arrow">→</span>
      <span class="pipe-node">Decision Tree</span>
      <span class="pipe-arrow">→</span>
      <span class="pipe-node active">Prediction</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Class distribution ───────────────────────────────────────────────────
    section("DISTRIBUSI KELAS (LABEL3)")
    vc = df["Label3"].value_counts().sort_index()
    total = len(df)
    cd1, cd2, cd3 = st.columns(3)

    for col, lbl, name, style in [
        (cd1, -1, "Down (-1)", "red"),
        (cd2,  0, "Neutral (0)", "yellow"),
        (cd3,  1, "Up (+1)", "green"),
    ]:
        count = vc.get(lbl, 0)
        pct   = count / total * 100
        color = LABEL_COLORS[lbl]
        col.markdown(
            f"""<div class="metric-card">
              <div class="metric-value" style="color:{color}; font-size:2.6rem;">{count}</div>
              <div class="metric-sub">{name}</div>
              <div class="metric-value" style="font-size:1.1rem; color:{color}; margin-top:6px;">{pct:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Price timeline ───────────────────────────────────────────────────────
    section("HARGA PASAR BULANAN")
    fig, ax = plt.subplots(figsize=(12, 3.2))
    colors_line = [LABEL_COLORS[l] for l in df["Label3"]]
    ax.plot(df["date"], df["monthly_average"], color=C_BORDER, lw=1.5, zorder=1)
    ax.scatter(df["date"], df["monthly_average"], c=colors_line, s=40, zorder=2)
    ax.set_xlabel("Tanggal", fontsize=8)
    ax.set_ylabel("Harga ($/bu)", fontsize=8)
    ax.grid(True, axis="y", linestyle="--")
    patches = [mpatches.Patch(color=LABEL_COLORS[k], label=LABEL_NAMES[k]) for k in [-1, 0, 1]]
    ax.legend(handles=patches, fontsize=7, loc="upper right",
              framealpha=0.2, facecolor=C_CARD, edgecolor=C_BORDER)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.markdown("## Exploratory Data Analysis")
    st.markdown(f"<span style='color:{C_MUTED}; font-size:0.82rem;'>Eksplorasi distribusi, tren, dan korelasi dalam dataset</span>", unsafe_allow_html=True)

    # ── Row 1: Price distribution + Label distribution
    section("DISTRIBUSI HARGA & LABEL")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df["monthly_average"], bins=18, color=C_ACCENT, alpha=0.75, edgecolor=C_BG)
        ax.axvline(df["monthly_average"].mean(), color=C_YELLOW, lw=1.5, linestyle="--",
                   label=f"Mean: {df['monthly_average'].mean():.2f}")
        ax.axvline(df["monthly_average"].median(), color=C_GREEN, lw=1.5, linestyle=":",
                   label=f"Median: {df['monthly_average'].median():.2f}")
        ax.set_xlabel("Harga Bulanan ($/bu)", fontsize=8)
        ax.set_ylabel("Frekuensi", fontsize=8)
        ax.set_title("Distribusi Harga Pasar", fontsize=9, color=C_TEXT)
        ax.legend(fontsize=7, framealpha=0.2, facecolor=C_CARD, edgecolor=C_BORDER)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with r1c2:
        vc = df["Label3"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(
            [LABEL_NAMES[k] for k in vc.index],
            vc.values,
            color=[LABEL_COLORS[k] for k in vc.index],
            width=0.55,
            edgecolor=C_BG,
        )
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    str(val), ha="center", va="bottom", fontsize=9, color=C_TEXT)
        ax.set_ylabel("Jumlah", fontsize=8)
        ax.set_title("Distribusi Kelas Label3", fontsize=9, color=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Row 2: Price per year + Label per year
    section("TREN PER TAHUN")
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        yearly = df.groupby("year")["monthly_average"].mean()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(yearly.index, yearly.values, color=C_ACCENT, marker="o", markersize=5, lw=2)
        ax.fill_between(yearly.index, yearly.values, alpha=0.12, color=C_ACCENT)
        ax.set_xlabel("Tahun", fontsize=8)
        ax.set_ylabel("Rata-rata Harga ($/bu)", fontsize=8)
        ax.set_title("Rata-rata Harga Per Tahun", fontsize=9, color=C_TEXT)
        ax.grid(True, axis="y", linestyle="--")
        for x, y in zip(yearly.index, yearly.values):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=C_MUTED)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with r2c2:
        pivot = df.groupby(["year", "Label3"]).size().unstack(fill_value=0)
        pivot = pivot.reindex(columns=[-1, 0, 1], fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        x = np.arange(len(pivot))
        w = 0.25
        for i, (lbl, clr) in enumerate([(-1, C_RED), (0, C_YELLOW), (1, C_GREEN)]):
            vals = pivot[lbl].values if lbl in pivot.columns else np.zeros(len(pivot))
            ax.bar(x + i*w, vals, w, color=clr, label=LABEL_NAMES[lbl], alpha=0.85)
        ax.set_xticks(x + w)
        ax.set_xticklabels(pivot.index, fontsize=7)
        ax.set_ylabel("Jumlah", fontsize=8)
        ax.set_title("Distribusi Label Per Tahun", fontsize=9, color=C_TEXT)
        ax.legend(fontsize=7, framealpha=0.2, facecolor=C_CARD, edgecolor=C_BORDER)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Boxplot price per label
    section("BOXPLOT HARGA PER KELAS SENTIMEN")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    groups = [df[df["Label3"] == k]["monthly_average"].values for k in [-1, 0, 1]]
    bp = ax.boxplot(groups, patch_artist=True, medianprops=dict(color=C_BG, lw=2))
    for patch, color in zip(bp["boxes"], [C_RED, C_YELLOW, C_GREEN]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([LABEL_NAMES[k] for k in [-1, 0, 1]], fontsize=8)
    ax.set_ylabel("Harga Bulanan ($/bu)", fontsize=8)
    ax.set_title("Distribusi Harga per Kelas Sentimen", fontsize=9, color=C_TEXT)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Stats table
    section("STATISTIK DESKRIPTIF")
    stats = df.groupby("Label3")["monthly_average"].describe().round(2)
    stats.index = [LABEL_NAMES[i] for i in stats.index]
    st.dataframe(stats, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Preprocessing":
    st.markdown("## Text Preprocessing")
    st.markdown(f"<span style='color:{C_MUTED}; font-size:0.82rem;'>Transformasi teks mentah → teks bersih siap diekstrak fiturnya</span>", unsafe_allow_html=True)

    # ── Pipeline steps ───────────────────────────────────────────────────────
    section("TAHAPAN PREPROCESSING")
    steps = [
        ("1. Lowercase", "Mengubah semua huruf menjadi lowercase", C_ACCENT),
        ("2. Remove Punctuation", "Menghapus tanda baca dan karakter khusus", C_ACCENT),
        ("3. Remove Numbers", "Menghapus angka yang tidak informatif", C_ACCENT),
        ("4. Stopword Removal", "Menghapus kata-kata umum (the, is, at, …)", C_ACCENT),
        ("5. Tokenization", "Memecah kalimat menjadi token kata", C_ACCENT),
        ("6. Cleaned Text", "Teks siap untuk ekstraksi fitur TF-IDF", C_GREEN),
    ]
    cols = st.columns(3)
    for i, (name, desc, color) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f"""<div class="metric-card" style="text-align:left; padding:14px;">
                  <div style="color:{color}; font-size:0.78rem; font-weight:700; margin-bottom:6px;">{name}</div>
                  <div style="color:{C_MUTED}; font-size:0.72rem;">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        if (i + 1) % 3 == 0 and i < len(steps) - 1:
            st.markdown("")

    # ── Before / After comparison ────────────────────────────────────────────
    section("PERBANDINGAN TEKS RAW vs CLEAN")
    idx = st.slider("Pilih record:", 0, len(df) - 1, 0)
    row = df.iloc[idx]

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown(f"<div class='section-header'>RAW TEXT</div>", unsafe_allow_html=True)
        st.markdown(
            f"""<div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px;
            padding:14px; font-size:0.78rem; line-height:1.7; color:{C_TEXT};'>{row['raw_text']}</div>""",
            unsafe_allow_html=True,
        )
        st.caption(f"📊 Word count: **{row['len_raw']}** kata")

    with cc2:
        st.markdown(f"<div class='section-header'>CLEAN TEXT</div>", unsafe_allow_html=True)
        st.markdown(
            f"""<div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px;
            padding:14px; font-size:0.78rem; line-height:1.7; color:{C_GREEN};'>{row['clean_text']}</div>""",
            unsafe_allow_html=True,
        )
        st.caption(f"📊 Word count: **{row['len_clean']}** kata")

    # ── Word count stats
    section("STATISTIK PANJANG TEKS")
    wc1, wc2, wc3 = st.columns(3)
    reduction = (1 - df["len_clean"].mean() / df["len_raw"].mean()) * 100
    wc1.markdown(metric_card("Avg Raw Words", f"{df['len_raw'].mean():.0f}", "kata per bulan"), unsafe_allow_html=True)
    wc2.markdown(metric_card("Avg Clean Words", f"{df['len_clean'].mean():.0f}", "kata per bulan", C_GREEN), unsafe_allow_html=True)
    wc3.markdown(metric_card("Reduksi", f"{reduction:.1f}%", "pengurangan noise", C_YELLOW), unsafe_allow_html=True)

    # ── Distribution of word lengths
    section("DISTRIBUSI PANJANG TEKS")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.2))
    ax1.hist(df["len_raw"],   bins=15, color=C_MUTED,  alpha=0.8, edgecolor=C_BG, label="Raw")
    ax1.hist(df["len_clean"], bins=15, color=C_ACCENT, alpha=0.8, edgecolor=C_BG, label="Clean")
    ax1.set_title("Overlay: Raw vs Clean", fontsize=9, color=C_TEXT)
    ax1.set_xlabel("Jumlah Kata", fontsize=8)
    ax1.legend(fontsize=7, framealpha=0.2, facecolor=C_CARD, edgecolor=C_BORDER)

    ax2.scatter(df["len_raw"], df["len_clean"],
                c=[LABEL_COLORS[l] for l in df["Label3"]], s=35, alpha=0.75)
    ax2.set_xlabel("Raw Word Count", fontsize=8)
    ax2.set_ylabel("Clean Word Count", fontsize=8)
    ax2.set_title("Korelasi Raw vs Clean Word Count", fontsize=9, color=C_TEXT)
    patches = [mpatches.Patch(color=LABEL_COLORS[k], label=LABEL_NAMES[k]) for k in [-1, 0, 1]]
    ax2.legend(handles=patches, fontsize=7, framealpha=0.2, facecolor=C_CARD, edgecolor=C_BORDER)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Extraction":
    st.markdown("## Feature Extraction")
    st.markdown(f"<span style='color:{C_MUTED}; font-size:0.82rem;'>TF-IDF dengan N-Grams (unigram + bigram) untuk representasi teks</span>", unsafe_allow_html=True)

    # ── TF-IDF info ──────────────────────────────────────────────────────────
    section("KONFIGURASI TF-IDF")
    fi1, fi2, fi3, fi4 = st.columns(4)
    fi1.markdown(metric_card("N-Gram Range", "(1, 2)", "unigram + bigram"), unsafe_allow_html=True)
    fi2.markdown(metric_card("Max Features", "500", "fitur teratas"), unsafe_allow_html=True)
    fi3.markdown(metric_card("Min DF", "2", "min dokumen"), unsafe_allow_html=True)
    fi4.markdown(metric_card("Vocab Size", str(len(model["vectorizer"].vocabulary_)), "unique n-grams", C_GREEN), unsafe_allow_html=True)

    # ── Top features ─────────────────────────────────────────────────────────
    section("TOP 20 FITUR TERPENTING (FEATURE IMPORTANCE)")
    top_feats = model["top_features"][:20]
    names  = [f[0] for f in top_feats]
    values = [f[1] for f in top_feats]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [C_ACCENT if v > np.mean(values) else C_MUTED for v in values]
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], alpha=0.85, height=0.65)
    ax.set_xlabel("Feature Importance", fontsize=8)
    ax.set_title("Feature Importance – Decision Tree + TF-IDF Bigrams", fontsize=9, color=C_TEXT)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=7, color=C_MUTED)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Top words frequency bar chart ────────────────────────────────────────
    section("FREKUENSI KATA TERBANYAK (CLEAN TEXT)")
    all_words = " ".join(df["clean_text"].dropna()).split()
    wf = Counter(all_words).most_common(25)
    words_  = [w[0] for w in wf]
    counts_ = [w[1] for w in wf]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(words_, counts_, color=C_ACCENT, alpha=0.75, edgecolor=C_BG)
    ax.set_xlabel("Kata", fontsize=8)
    ax.set_ylabel("Frekuensi", fontsize=8)
    ax.set_title("25 Kata Paling Sering Muncul", fontsize=9, color=C_TEXT)
    plt.xticks(rotation=40, ha="right", fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Per-class top words ───────────────────────────────────────────────────
    section("KATA DOMINAN PER KELAS")
    lc1, lc2, lc3 = st.columns(3)
    for col, label, lname, color in [
        (lc1, -1, "Down (-1)", C_RED),
        (lc2,  0, "Neutral (0)", C_YELLOW),
        (lc3,  1, "Up (+1)", C_GREEN),
    ]:
        subset = df[df["Label3"] == label]["clean_text"].dropna()
        words = " ".join(subset).split()
        top10 = Counter(words).most_common(10)
        with col:
            st.markdown(f"<div class='section-header' style='color:{color};'>{lname}</div>", unsafe_allow_html=True)
            for w, c in top10:
                pct = c / len(words) * 100
                st.markdown(
                    f"""<div style='display:flex; justify-content:space-between; font-size:0.75rem;
                    border-bottom:1px solid {C_BORDER}; padding:4px 0;'>
                    <span style='color:{C_TEXT};'>{w}</span>
                    <span style='color:{color};'>{c} ({pct:.1f}%)</span></div>""",
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Evaluation":
    st.markdown("## Model Evaluation")
    st.markdown(f"<span style='color:{C_MUTED}; font-size:0.82rem;'>Performa Decision Tree + TF-IDF N-Grams pada test set</span>", unsafe_allow_html=True)

    acc = model["accuracy"]
    cr  = model["classification_report"]

    # ── Top metrics ──────────────────────────────────────────────────────────
    section("METRIK UTAMA")
    me1, me2, me3, me4 = st.columns(4)
    me1.markdown(metric_card("Accuracy", f"{acc*100:.2f}%", "test set", C_GREEN), unsafe_allow_html=True)
    me2.markdown(metric_card("CV Mean", f"{model['cv_scores'].mean()*100:.2f}%", "5-fold stratified", C_ACCENT), unsafe_allow_html=True)
    me3.markdown(metric_card("CV Std", f"±{model['cv_scores'].std()*100:.2f}%", "variance", C_YELLOW), unsafe_allow_html=True)
    me4.markdown(metric_card("Test Size", "17", "records (20%)", C_MUTED), unsafe_allow_html=True)

    # ── Confusion matrix + Classification report ──────────────────────────────
    section("CONFUSION MATRIX & CLASSIFICATION REPORT")
    col_cm, col_cr = st.columns([1, 1.2])

    with col_cm:
        cm = model["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="YlOrRd", aspect="auto", vmin=0)
        labels = ["Down(-1)", "Neutral(0)", "Up(+1)"]
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
        ax.set_title("Confusion Matrix", fontsize=9, color=C_TEXT)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="black" if cm[i,j] > cm.max()/2 else C_TEXT)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_cr:
        rows = []
        for k, name in [("Down(-1)", "Down (-1)"), ("Neutral(0)", "Neutral (0)"), ("Up(+1)", "Up (+1)")]:
            if k in cr:
                rows.append({
                    "Kelas": name,
                    "Precision": f"{cr[k]['precision']:.2f}",
                    "Recall":    f"{cr[k]['recall']:.2f}",
                    "F1-Score":  f"{cr[k]['f1-score']:.2f}",
                    "Support":   int(cr[k]["support"]),
                })
        cdf = pd.DataFrame(rows)
        st.dataframe(cdf, use_container_width=True, hide_index=True)

        ma = cr.get("macro avg", {})
        wa = cr.get("weighted avg", {})
        st.markdown(
            f"""<div style='margin-top:12px; display:flex; gap:10px; flex-wrap:wrap;'>
            <div class='metric-card' style='flex:1;'>
              <div class='metric-label'>Macro Avg F1</div>
              <div class='metric-value' style='color:{C_ACCENT}; font-size:1.5rem;'>{ma.get('f1-score',0):.3f}</div>
            </div>
            <div class='metric-card' style='flex:1;'>
              <div class='metric-label'>Weighted Avg F1</div>
              <div class='metric-value' style='color:{C_GREEN}; font-size:1.5rem;'>{wa.get('f1-score',0):.3f}</div>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Cross-validation scores ───────────────────────────────────────────────
    section("CROSS-VALIDATION SCORES (5-FOLD)")
    cv_scores = model["cv_scores"]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    colors = [C_GREEN if s >= cv_scores.mean() else C_RED for s in cv_scores]
    bars = ax.bar(folds, cv_scores, color=colors, alpha=0.85, width=0.5)
    ax.axhline(cv_scores.mean(), color=C_YELLOW, linestyle="--", lw=1.5,
               label=f"Mean: {cv_scores.mean():.3f}")
    for bar, val in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.legend(fontsize=8, framealpha=0.2, facecolor=C_CARD, edgecolor=C_BORDER)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Model comparison (3 models) ───────────────────────────────────────────
    section("PERBANDINGAN MODEL")
    X, y = df["clean_text"], df["Label3"]
    tfidf2 = TfidfVectorizer(ngram_range=(1, 2), max_features=500, min_df=2)
    X2 = tfidf2.fit_transform(X)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X2, y, test_size=0.2, random_state=7, stratify=y)

    model_scores = {}
    for name, clf in [
        ("Decision Tree", DecisionTreeClassifier(max_depth=8, min_samples_split=4, random_state=42)),
        ("Naive Bayes",   MultinomialNB()),
        ("Linear SVM",    LinearSVC(max_iter=1000, random_state=42)),
    ]:
        clf.fit(X_tr2, y_tr2)
        model_scores[name] = accuracy_score(y_te2, clf.predict(X_te2))

    fig, ax = plt.subplots(figsize=(8, 3))
    bar_colors = [C_GREEN if v == max(model_scores.values()) else C_ACCENT for v in model_scores.values()]
    bars = ax.barh(list(model_scores.keys()), list(model_scores.values()),
                   color=bar_colors, alpha=0.85, height=0.45)
    for bar, val in zip(bars, model_scores.values()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy", fontsize=8)
    ax.set_title("Perbandingan Accuracy – Test Set", fontsize=9, color=C_TEXT)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.markdown("## Sentiment Prediction")
    st.markdown(f"<span style='color:{C_MUTED}; font-size:0.82rem;'>Masukkan judul berita soybean untuk diprediksi sentimennya</span>", unsafe_allow_html=True)

    section("INPUT TEKS BERITA")
    input_text = st.text_area(
        "Masukkan headline / ringkasan berita:",
        placeholder="Contoh: Brazil soybean export record high while Argentina farmers hold stocks due to currency uncertainty...",
        height=120,
        label_visibility="collapsed",
    )

    # ── Sample buttons ────────────────────────────────────────────────────────
    st.markdown("<div style='font-size:0.72rem; color:{C_MUTED}; margin-bottom:6px;'>Atau coba contoh:</div>".replace("{C_MUTED}", C_MUTED), unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    if sc1.button("📈 Contoh Positif"):
        input_text = "brazil soybean export record high farmer selling strong demand china positive outlook production increase"
    if sc2.button("📉 Contoh Negatif"):
        input_text = "dry weather impacts soybean yields concern disease spread harvest delay production estimate downgraded"
    if sc3.button("➖ Contoh Netral"):
        input_text = "usda report weekly soybean market update crop condition slight change brazil argentina acreage"

    if st.button("🔍 PREDIKSI SENTIMEN", use_container_width=True):
        if not input_text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            # Clean input
            clean = re.sub(r"[^a-zA-Z\s]", " ", input_text.lower())
            clean = re.sub(r"\s+", " ", clean).strip()

            vec_input = model["vectorizer"].transform([clean])
            pred      = model["model"].predict(vec_input)[0]
            proba     = model["model"].predict_proba(vec_input)[0]

            lbl_map = {-1: ("📉 DOWN", C_RED, "pill-red"), 0: ("➖ NEUTRAL", C_YELLOW, "pill-yellow"), 1: ("📈 UP", C_GREEN, "pill-green")}
            icon, color, pill_class = lbl_map[pred]

            st.markdown(
                f"""<div class='pred-box' style='border:2px solid {color}; background:rgba(0,0,0,0.3);'>
                  <div style='font-size:0.72rem; color:{C_MUTED}; margin-bottom:8px;'>PREDIKSI SENTIMEN</div>
                  <div style='font-size:2.8rem; font-weight:700; color:{color};'>{icon}</div>
                  <div style='font-size:1rem; color:{C_MUTED}; margin-top:4px;'>{LABEL_NAMES[pred]}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Probability bars
            section("PROBABILITAS PER KELAS")
            classes = model["model"].classes_
            for cls, prob in zip(classes, proba):
                clr = LABEL_COLORS[cls]
                st.markdown(
                    f"""<div style='display:flex; align-items:center; gap:12px; margin:6px 0;'>
                    <div style='width:90px; font-size:0.75rem; color:{clr};'>{LABEL_NAMES[cls]}</div>
                    <div style='flex:1; background:{C_BORDER}; border-radius:4px; height:18px; overflow:hidden;'>
                      <div style='width:{prob*100:.1f}%; height:100%; background:{clr}; border-radius:4px;
                      transition:width 0.5s;'></div>
                    </div>
                    <div style='width:50px; font-size:0.78rem; color:{C_TEXT}; text-align:right;'>{prob*100:.1f}%</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Cleaned text
            section("TEKS SETELAH CLEANING")
            st.markdown(
                f"""<div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px;
                padding:12px; font-size:0.78rem; color:{C_GREEN};'>{clean}</div>""",
                unsafe_allow_html=True,
            )

    # ── Batch prediction from dataset ─────────────────────────────────────────
    section("BATCH PREDICTION – SAMPEL DATASET")
    sample_n = st.slider("Tampilkan n record dari dataset:", 5, 30, 10)
    sample_df = df.sample(sample_n, random_state=42).copy()
    vecs = model["vectorizer"].transform(sample_df["clean_text"])
    sample_df["PREDICTED"] = model["model"].predict(vecs)
    sample_df["CORRECT"]   = (sample_df["PREDICTED"] == sample_df["Label3"])

    out = sample_df[["date", "monthly_average", "Label3", "PREDICTED", "CORRECT"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.columns = ["Date", "Avg Price", "Actual", "Predicted", "Correct"]
    st.dataframe(out, use_container_width=True, hide_index=True)

    batch_acc = out["Correct"].mean() * 100
    st.markdown(
        f"<div style='font-size:0.82rem; color:{C_GREEN}; margin-top:8px;'>✅ Batch Accuracy: <strong>{batch_acc:.1f}%</strong> dari {sample_n} record</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CONCLUSION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Conclusion":
    st.markdown("## Conclusion")
    st.markdown(f"<span style='color:{C_MUTED}; font-size:0.82rem;'>Ringkasan temuan, limitasi, dan rekomendasi pengembangan</span>", unsafe_allow_html=True)

    # ── Summary cards ────────────────────────────────────────────────────────
    section("RINGKASAN PROYEK")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown(
            f"""<div class='metric-card' style='text-align:left;'>
            <div style='color:{C_ACCENT}; font-weight:700; font-size:0.85rem; margin-bottom:10px;'>🎯 Tujuan Penelitian</div>
            <div style='color:{C_TEXT}; font-size:0.8rem; line-height:1.8;'>
            Membangun sistem klasifikasi sentimen berita komoditas soybean untuk memprediksi arah pergerakan harga pasar bulanan
            menggunakan pendekatan NLP berbasis TF-IDF + N-Grams dan Decision Tree.
            </div></div>""",
            unsafe_allow_html=True,
        )
    with cc2:
        st.markdown(
            f"""<div class='metric-card' style='text-align:left;'>
            <div style='color:{C_GREEN}; font-weight:700; font-size:0.85rem; margin-bottom:10px;'>✅ Hasil Utama</div>
            <div style='color:{C_TEXT}; font-size:0.8rem; line-height:1.8;'>
            Model terbaik: Decision Tree + TF-IDF Bigrams dengan akurasi
            <span style='color:{C_GREEN}; font-weight:700;'>{model['accuracy']*100:.2f}%</span> pada test set.
            Sentimen <em>Down</em> paling banyak ({df[df['Label3']==-1].shape[0]} record, 41%).
            Kata kunci dominan: <em>soybean, brazil, corn, mato, price</em>.
            </div></div>""",
            unsafe_allow_html=True,
        )

    # ── Findings ──────────────────────────────────────────────────────────────
    section("TEMUAN KUNCI")
    findings = [
        ("📊", "Dataset Kecil", f"Hanya 83 record (Jan 2014 – Des 2020), satu record per bulan. Dataset yang lebih besar akan meningkatkan performa model secara signifikan.", C_MUTED),
        ("📰", "Kualitas Headline", "Rata-rata 10 headline per bulan digabung sebagai representasi sentimen bulanan. Agregasi ini bisa kehilangan nuansa sentimen individual.", C_MUTED),
        ("🌿", "Dominasi Tema", "Kata soybean, brazil, dan corn mendominasi seluruh kelas, menunjukkan berita bersifat sangat domain-spesifik dengan noise rendah.", C_MUTED),
        ("📉", "Ketidakseimbangan Kelas", f"Down: 41%, Up: 39.8%, Neutral: 19.3% – kelas Neutral paling sedikit dan paling sulit diprediksi oleh model.", C_YELLOW),
        ("💡", "TF-IDF + Bigram", "Penggunaan bigram (2-gram) terbukti lebih informatif daripada unigram saja karena menangkap frasa kontekstual seperti 'soybean harvest', 'corn price'.", C_GREEN),
        ("🌍", "Faktor Eksternal", "Harga soybean sangat dipengaruhi faktor geopolitik (kebijakan ekspor, cuaca, kurs) yang sulit ditangkap hanya dari teks headline.", C_RED),
    ]
    for i in range(0, len(findings), 2):
        fc1, fc2 = st.columns(2)
        for col, item in [(fc1, findings[i]), (fc2, findings[i+1] if i+1 < len(findings) else None)]:
            if item:
                icon, title, desc, clr = item
                col.markdown(
                    f"""<div class='metric-card' style='text-align:left; margin-bottom:12px;'>
                    <div style='color:{clr}; font-size:0.82rem; font-weight:700; margin-bottom:6px;'>{icon} {title}</div>
                    <div style='color:{C_MUTED}; font-size:0.75rem; line-height:1.7;'>{desc}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ── Limitations & recommendations ─────────────────────────────────────────
    section("REKOMENDASI PENGEMBANGAN")
    recs = [
        ("🔢", "Perbesar Dataset", "Tambahkan lebih banyak sumber berita (Reuters, Bloomberg, AgWeb) dan perpanjang periode waktu hingga 2024 untuk dataset yang lebih representatif."),
        ("🤖", "Deep Learning", "Eksplorasi BERT, FinBERT, atau model Transformer yang pre-trained pada domain keuangan/agrikultur untuk representasi teks yang lebih kaya."),
        ("📈", "Multi-modal Features", "Gabungkan fitur teks dengan data teknikal harga (moving average, RSI, volume) sebagai fitur tambahan untuk prediksi yang lebih robust."),
        ("⚖️", "Tangani Imbalanced Class", "Terapkan teknik SMOTE, class weighting, atau threshold tuning untuk meningkatkan performa pada kelas Neutral yang minoritas."),
        ("🔄", "Ensemble Methods", "Coba Random Forest, XGBoost, atau Voting Classifier untuk mengurangi variance dan meningkatkan generalisasi model."),
        ("🌐", "Real-time Pipeline", "Bangun pipeline real-time yang mengambil berita terbaru, membersihkan, dan memprediksi sentimen secara otomatis setiap bulan."),
    ]
    for i in range(0, len(recs), 3):
        rc1, rc2, rc3 = st.columns(3)
        for col, item in [(rc1, recs[i]), (rc2, recs[i+1] if i+1 < len(recs) else None), (rc3, recs[i+2] if i+2 < len(recs) else None)]:
            if item:
                icon, title, desc = item
                col.markdown(
                    f"""<div class='metric-card' style='text-align:left; margin-bottom:12px;'>
                    <div style='color:{C_ACCENT}; font-size:0.82rem; font-weight:700; margin-bottom:6px;'>{icon} {title}</div>
                    <div style='color:{C_MUTED}; font-size:0.74rem; line-height:1.7;'>{desc}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ── Final accuracy recap ──────────────────────────────────────────────────
    section("SCORECARD AKHIR")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.markdown(metric_card("Total Data",     "83 records",    "Jan 2014 – Des 2020"), unsafe_allow_html=True)
    sc2.markdown(metric_card("Model",          "DT + Bigrams",  "TF-IDF N-Grams",      C_ACCENT), unsafe_allow_html=True)
    sc3.markdown(metric_card("Test Accuracy",  f"{model['accuracy']*100:.2f}%", "test set", C_GREEN), unsafe_allow_html=True)
    sc4.markdown(metric_card("CV Mean Acc",    f"{model['cv_scores'].mean()*100:.2f}%", "5-fold", C_YELLOW), unsafe_allow_html=True)
    sc5.markdown(metric_card("Vocab Size",     str(len(model["vectorizer"].vocabulary_)), "unique n-grams", C_MUTED), unsafe_allow_html=True)

    st.markdown("<hr class='custom-hr'/>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center; color:{C_MUTED}; font-size:0.7rem; margin-top:8px;'>"
        "SOYSIGNAL v1.0 · NLP Sentiment Analysis · Soybean Commodity Market · 2014–2020"
        "</div>",
        unsafe_allow_html=True,
    )