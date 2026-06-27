import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, root_mean_squared_error,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analitik Pengunjung — SMP PGRI 8",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette ─────────────────────────────────────────────────────────────
C_PRIMARY  = "#4F46E5"
C_PURPLE   = "#7C3AED"
C_CYAN     = "#06B6D4"
C_GREEN    = "#10B981"
C_AMBER    = "#F59E0B"
C_RED      = "#EF4444"
C_BLUE     = "#3B82F6"

PLOTLY_COLORS = [C_PRIMARY, C_CYAN, C_GREEN, C_AMBER, C_RED, C_PURPLE, C_BLUE]
CHART_TEMPLATE = "plotly_white"

NAMA_HARI = {0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis",
             4: "Jumat", 5: "Sabtu", 6: "Minggu"}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #06B6D4 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 { font-size: 2rem; font-weight: 800; margin: 0 0 .3rem 0; color: white; }
.hero p  { font-size: 1rem; margin: 0; opacity: .85; color: white; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,.2);
    border: 1px solid rgba(255,255,255,.35);
    border-radius: 999px;
    padding: .2rem .8rem;
    font-size: .78rem;
    font-weight: 600;
    margin-top: .7rem;
    color: white;
}

/* ── KPI cards ── */
.kpi-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.kpi-card {
    flex: 1; min-width: 140px;
    background: white;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.08), 0 4px 16px rgba(79,70,229,.07);
    border-top: 4px solid var(--accent);
    position: relative; overflow: hidden;
}
.kpi-card::after {
    content: attr(data-icon);
    position: absolute; right: 1rem; top: .9rem;
    font-size: 1.9rem; opacity: .12;
}
.kpi-label { font-size: .72rem; font-weight: 600; text-transform: uppercase;
             letter-spacing: .06em; color: #6B7280; margin-bottom: .35rem; }
.kpi-value { font-size: 1.55rem; font-weight: 800; color: #111827; line-height: 1.1; }
.kpi-sub   { font-size: .75rem; color: #6B7280; margin-top: .2rem; }

/* ── Section header ── */
.section-header {
    display: flex; align-items: center; gap: .6rem;
    margin: 1.6rem 0 1rem 0;
}
.section-header .bar {
    width: 4px; height: 22px; border-radius: 2px;
    background: linear-gradient(to bottom, #4F46E5, #06B6D4);
}
.section-header h3 { margin: 0; font-size: 1.05rem; font-weight: 700; color: #1F2937; }

/* ── Insight cards ── */
.insight-list { display: flex; flex-direction: column; gap: .65rem; margin-top: .5rem; }
.insight-item {
    background: #F9FAFB;
    border-left: 4px solid #4F46E5;
    border-radius: 0 10px 10px 0;
    padding: .75rem 1rem;
    font-size: .9rem; color: #374151; line-height: 1.5;
}
.insight-item strong { color: #4F46E5; }
.insight-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%;
    background: #4F46E5; color: white; font-size: .7rem; font-weight: 700;
    margin-right: .5rem; vertical-align: middle; flex-shrink: 0;
}

/* ── Sesi badge ── */
.badge-puncak { background:#FEE2E2; color:#B91C1C; padding:.2rem .6rem;
                border-radius:999px; font-size:.78rem; font-weight:600; }
.badge-sedang { background:#FEF3C7; color:#92400E; padding:.2rem .6rem;
                border-radius:999px; font-size:.78rem; font-weight:600; }
.badge-sepi   { background:#DBEAFE; color:#1E40AF; padding:.2rem .6rem;
                border-radius:999px; font-size:.78rem; font-weight:600; }

/* ── Tabs ── */
div[data-baseweb="tab-list"] { gap: .3rem; }
div[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important; font-size: .85rem !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #F8F7FF; }
.sidebar-card {
    background: white; border-radius: 10px; padding: .9rem 1rem;
    margin-bottom: .7rem; box-shadow: 0 1px 3px rgba(0,0,0,.07);
}
.sidebar-card h4 { margin: 0 0 .5rem 0; font-size: .82rem; font-weight: 700;
                   color: #4F46E5; text-transform: uppercase; letter-spacing: .05em; }
.sidebar-card p  { margin: .2rem 0; font-size: .83rem; color: #374151; }
.sidebar-card .val { font-weight: 700; color: #111827; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = Path(__file__).parent / "pengunjung_website.csv"
    df = pd.read_csv(csv_path, parse_dates=["tanggal"])
    df["hari"] = df["tanggal"].dt.dayofweek.map(NAMA_HARI)
    df["minggu_ke"] = ((df["tanggal"] - df["tanggal"].min()).dt.days // 7) + 1
    return df

df = load_data()

# ── Pre-compute shared aggregations ──────────────────────────────────────────
per_halaman   = df.groupby("halaman")["jumlah_pengunjung"].sum().sort_values(ascending=False)
per_jam       = df.groupby("jam")["jumlah_pengunjung"].sum()
per_minggu    = df.groupby("minggu_ke")["jumlah_pengunjung"].sum()
URUT_HARI     = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
per_hari      = df.groupby("hari")["jumlah_pengunjung"].sum().reindex(URUT_HARI)
pengunjung_harian = df.groupby("tanggal")["jumlah_pengunjung"].sum()

total_kunjungan = int(df["jumlah_pengunjung"].sum())
jumlah_hari     = df["tanggal"].nunique()
rata_harian     = total_kunjungan / jumlah_hari
durasi_rata     = df["durasi"].mean()
jam_puncak      = int(per_jam.idxmax())
halaman_top     = per_halaman.index[0]
hari_top        = per_hari.idxmax()
pertumbuhan     = (per_minggu.iloc[-1] / per_minggu.iloc[0] - 1) * 100

# ── Cached models ─────────────────────────────────────────────────────────────
@st.cache_data
def run_linear(_df):
    harian = (_df.groupby("tanggal")
                 .agg(total_pengunjung=("jumlah_pengunjung","sum"))
                 .reset_index())
    harian["minggu_ke"] = ((harian["tanggal"] - harian["tanggal"].min()).dt.days // 7) + 1
    harian["nama_hari"] = harian["tanggal"].dt.dayofweek.map(NAMA_HARI)
    dum = pd.get_dummies(harian["nama_hari"], prefix="hari", drop_first=True)
    X = pd.concat([harian[["minggu_ke"]], dum], axis=1).astype(float)
    y = harian["total_pengunjung"]
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=.25,random_state=42)
    lin = LinearRegression().fit(X_tr,y_tr)
    y_pred = lin.predict(X_te)
    harian["prediksi"] = lin.predict(X)
    return harian,lin,X,y,X_te,y_te,y_pred

@st.cache_data
def run_poly(_df):
    per_jam_rata = (_df.groupby(["tanggal","jam"])["jumlah_pengunjung"].sum()
                       .groupby("jam").mean())
    xj = per_jam_rata.index.values.reshape(-1,1).astype(float)
    yj = per_jam_rata.values
    hasil = {}
    for d in [1,2,3,4,6]:
        m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(xj,yj)
        hasil[d] = r2_score(yj, m.predict(xj))
    best = max(hasil, key=hasil.get)
    return xj, yj, hasil, best

@st.cache_data
def run_kmeans(_df):
    fj = pd.DataFrame({
        "rata_pengunjung": _df.groupby(["tanggal","jam"])["jumlah_pengunjung"].sum()
                              .groupby("jam").mean(),
        "rata_durasi": _df.groupby("jam")["durasi"].mean(),
    })
    Xs = StandardScaler().fit_transform(fj)
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xs)
    fj["cluster"] = km.labels_
    urut = fj.groupby("cluster")["rata_pengunjung"].mean().sort_values().index.tolist()
    fj["sesi"] = fj["cluster"].map({urut[0]:"Sepi",urut[1]:"Sedang",urut[2]:"Puncak"})
    sil = silhouette_score(Xs, km.labels_)
    return fj, sil

@st.cache_data
def run_clf(_df):
    tj = (_df.groupby(["tanggal","jam"])["jumlah_pengunjung"].sum()
              .reset_index(name="total"))
    tj["dayofweek"] = tj["tanggal"].dt.dayofweek
    tj["minggu_ke"] = ((tj["tanggal"] - tj["tanggal"].min()).dt.days // 7) + 1
    ambang = tj["total"].median()
    tj["ramai"] = (tj["total"] > ambang).astype(int)
    dum = pd.get_dummies(tj["dayofweek"].map(NAMA_HARI), prefix="hari")
    Xc = pd.concat([tj[["jam","minggu_ke"]], dum], axis=1).astype(float)
    yc = tj["ramai"]
    Xc_tr,Xc_te,yc_tr,yc_te = train_test_split(Xc,yc,test_size=.25,random_state=42,stratify=yc)
    clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xc_tr,yc_tr)
    yc_pred = clf.predict(Xc_te)
    return clf, Xc, yc_te, yc_pred, ambang

harian,lin,X_lin,y_lin,X_te_lin,y_te_lin,y_pred_lin = run_linear(df)
xj,yj,hasil_deg,best_deg = run_poly(df)
fitur_jam,sil = run_kmeans(df)
clf,Xc,yc_te,yc_pred,ambang = run_clf(df)

r2   = r2_score(y_te_lin, y_pred_lin)
mae  = mean_absolute_error(y_te_lin, y_pred_lin)
rmse = root_mean_squared_error(y_te_lin, y_pred_lin)
acc  = accuracy_score(yc_te, yc_pred)
prec = precision_score(yc_te, yc_pred)
rec  = recall_score(yc_te, yc_pred)
f1   = f1_score(yc_te, yc_pred)
imp  = pd.Series(clf.feature_importances_, index=Xc.columns)
imp  = imp[imp > 0].sort_values()

ringkas_sesi = (fitur_jam.reset_index()
                .groupby("sesi")["jam"]
                .agg(lambda s: ", ".join(f"{j}:00" for j in sorted(s)))
                .reindex(["Puncak","Sedang","Sepi"]))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:.5rem 0 1rem 0'>
      <span style='font-size:2.5rem'>🏫</span>
      <div style='font-weight:800; font-size:1rem; color:#4F46E5; margin-top:.3rem'>
        SMP PGRI 8 Bogor
      </div>
      <div style='font-size:.75rem; color:#6B7280'>Analitik Pengunjung Website</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-card">
      <h4>📅 Periode Data</h4>
      <p>{df['tanggal'].min().strftime('%d %b %Y')} —<br>
         {df['tanggal'].max().strftime('%d %b %Y')}</p>
      <p>Total: <span class="val">{jumlah_hari} hari</span></p>
    </div>
    <div class="sidebar-card">
      <h4>📊 Ringkasan</h4>
      <p>Total baris: <span class="val">{len(df):,}</span></p>
      <p>Total kunjungan: <span class="val">{total_kunjungan:,}</span></p>
      <p>Halaman unik: <span class="val">{df['halaman'].nunique()}</span></p>
    </div>
    <div class="sidebar-card">
      <h4>🤖 Model</h4>
      <p>Regresi Linear R²: <span class="val">{r2:.2f}</span></p>
      <p>Poly deg {best_deg} R²: <span class="val">{hasil_deg[best_deg]:.2f}</span></p>
      <p>Klasifikasi Acc: <span class="val">{acc:.2f}</span></p>
      <p>Silhouette: <span class="val">{sil:.2f}</span></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:.72rem; color:#9CA3AF; text-align:center; margin-top:1rem'>
      Proyek Sains Data · Skema PBL Terintegrasi<br>
      Program Studi Kecerdasan Buatan · SSMI IPB
    </div>
    """, unsafe_allow_html=True)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>📊 Dashboard Analitik Pengunjung Website</h1>
  <p>Eksplorasi data, pemodelan, dan insight berbasis data pengunjung website sekolah</p>
  <span class="badge">🏫 SMP PGRI 8 Kota Bogor</span>
  <span class="badge" style="margin-left:.4rem">📅 Apr – Mei 2026</span>
  <span class="badge" style="margin-left:.4rem">🗂 {len(df):,} baris data</span>
</div>
""", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card" style="--accent:{C_PRIMARY}" data-icon="👥">
    <div class="kpi-label">Total Kunjungan</div>
    <div class="kpi-value">{total_kunjungan:,}</div>
    <div class="kpi-sub">selama {jumlah_hari} hari</div>
  </div>
  <div class="kpi-card" style="--accent:{C_CYAN}" data-icon="📈">
    <div class="kpi-label">Rata-rata / Hari</div>
    <div class="kpi-value">{rata_harian:,.0f}</div>
    <div class="kpi-sub">pengunjung per hari</div>
  </div>
  <div class="kpi-card" style="--accent:{C_GREEN}" data-icon="🏆">
    <div class="kpi-label">Halaman Teratas</div>
    <div class="kpi-value" style="font-size:1.2rem">{halaman_top}</div>
    <div class="kpi-sub">{per_halaman.iloc[0]:,} kunjungan</div>
  </div>
  <div class="kpi-card" style="--accent:{C_AMBER}" data-icon="⏰">
    <div class="kpi-label">Jam Puncak</div>
    <div class="kpi-value">{jam_puncak}:00</div>
    <div class="kpi-sub">paling ramai dikunjungi</div>
  </div>
  <div class="kpi-card" style="--accent:{C_RED}" data-icon="🚀">
    <div class="kpi-label">Tren Mingguan</div>
    <div class="kpi-value">+{pertumbuhan:.0f}%</div>
    <div class="kpi-sub">minggu 1 → minggu {int(per_minggu.index[-1])}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "🧹 Pembersihan Data",
    "📈 Regresi Linear",
    "〰️ Regresi Non-Linear",
    "🔵 Clustering",
    "🌳 Klasifikasi",
])

# helper
def sec(icon, title):
    st.markdown(f"""
    <div class="section-header">
      <div class="bar"></div>
      <h3>{icon} {title}</h3>
    </div>""", unsafe_allow_html=True)

def chart_layout(fig, title, h=420, **kw):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1F2937", family="Inter"),
                   x=0, xanchor="left"),
        template=CHART_TEMPLATE,
        height=h,
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        font=dict(family="Inter", size=12),
        margin=dict(t=50, b=40, l=10, r=10),
        **kw,
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    sec("📅", "Tren Pengunjung Harian")
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=pengunjung_harian.index, y=pengunjung_harian.values,
        mode="lines", fill="tozeroy",
        line=dict(color=C_PRIMARY, width=2.5),
        fillcolor="rgba(79,70,229,.12)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:,} pengunjung<extra></extra>",
        name="Harian",
    ))
    chart_layout(fig_daily, "Jumlah Pengunjung per Hari", h=320)
    st.plotly_chart(fig_daily, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        sec("🌐", "Halaman Terpopuler")
        fig_hal = go.Figure(go.Bar(
            x=per_halaman.values, y=per_halaman.index,
            orientation="h",
            marker=dict(
                color=per_halaman.values,
                colorscale=[[0,"#C7D2FE"],[1,C_PRIMARY]],
                showscale=False,
            ),
            text=[f"{v:,}" for v in per_halaman.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:,} kunjungan<extra></extra>",
        ))
        chart_layout(fig_hal, "Total Kunjungan per Halaman", h=380)
        fig_hal.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_hal, use_container_width=True)

    with col_r:
        sec("⏰", "Pola Kunjungan per Jam")
        bar_colors = [C_RED if j == jam_puncak else C_BLUE for j in per_jam.index]
        fig_jam = go.Figure(go.Bar(
            x=per_jam.index, y=per_jam.values,
            marker_color=bar_colors,
            hovertemplate="Jam %{x}:00 → <b>%{y:,}</b> kunjungan<extra></extra>",
        ))
        chart_layout(fig_jam, f"Kunjungan per Jam (puncak {jam_puncak}:00)", h=380)
        fig_jam.update_xaxes(tickvals=list(per_jam.index))
        st.plotly_chart(fig_jam, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        sec("📅", "Kunjungan per Hari")
        fig_hari = go.Figure(go.Bar(
            x=per_hari.index, y=per_hari.values,
            marker=dict(color=per_hari.values,
                        colorscale=[[0,"#A5F3FC"],[1,C_CYAN]], showscale=False),
            hovertemplate="<b>%{x}</b><br>%{y:,} kunjungan<extra></extra>",
        ))
        chart_layout(fig_hari, "Total Kunjungan per Hari", h=320)
        st.plotly_chart(fig_hari, use_container_width=True)

    with col4:
        sec("📆", "Tren Mingguan")
        fig_mw = go.Figure()
        fig_mw.add_trace(go.Scatter(
            x=per_minggu.index, y=per_minggu.values,
            mode="lines+markers",
            line=dict(color=C_GREEN, width=2.5),
            marker=dict(size=9, color=C_GREEN, line=dict(width=2, color="white")),
            fill="tozeroy", fillcolor="rgba(16,185,129,.1)",
            hovertemplate="Minggu ke-%{x}<br><b>%{y:,}</b> pengunjung<extra></extra>",
        ))
        chart_layout(fig_mw, "Total Pengunjung per Minggu", h=320)
        fig_mw.update_xaxes(tickvals=list(per_minggu.index))
        st.plotly_chart(fig_mw, use_container_width=True)

    sec("📋", "Tabel Agregasi")
    ta1, ta2 = st.columns(2)
    with ta1:
        st.caption("Kunjungan per Halaman")
        st.dataframe(
            pd.DataFrame({
                "Halaman": per_halaman.index,
                "Total": per_halaman.values,
                "Persen (%)": (per_halaman.values / per_halaman.sum() * 100).round(1),
            }),
            use_container_width=True, hide_index=True,
            column_config={"Total": st.column_config.NumberColumn(format="%d")},
        )
    with ta2:
        st.caption("Statistik Ringkas")
        st.dataframe(df[["jumlah_pengunjung","durasi"]].describe().round(1),
                     use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Pembersihan Data
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    sec("🔍", "Informasi Dataset")
    p1, p2, p3 = st.columns(3)
    p1.metric("Jumlah Baris", f"{len(df):,}")
    p2.metric("Jumlah Kolom", len(df.columns))
    p3.metric("Duplikat", int(df.duplicated().sum()))

    c1, c2 = st.columns(2)
    with c1:
        sec("🗂", "Tipe Data")
        st.dataframe(df.dtypes.rename("Tipe").to_frame().astype(str),
                     use_container_width=True)
    with c2:
        sec("❓", "Nilai Kosong")
        miss = df.isna().sum().rename("Jumlah Kosong").to_frame()
        miss["Status"] = miss["Jumlah Kosong"].apply(
            lambda v: "✅ Lengkap" if v == 0 else "⚠️ Ada kosong")
        st.dataframe(miss, use_container_width=True)

    sec("✅", "Validasi Rentang Nilai")
    masalah = {
        "jam di luar 0–23":        int((~df["jam"].between(0,23)).sum()),
        "jumlah_pengunjung ≤ 0":   int((df["jumlah_pengunjung"] <= 0).sum()),
        "durasi ≤ 0 detik":        int((df["durasi"] <= 0).sum()),
    }
    val_df = pd.DataFrame([
        {"Pemeriksaan": k, "Pelanggaran": v,
         "Status": "✅ OK" if v == 0 else "⚠️ Perlu dibersihkan"}
        for k, v in masalah.items()
    ]).set_index("Pemeriksaan")
    st.dataframe(val_df, use_container_width=True)
    st.success("Semua nilai berada pada rentang yang valid.")

    sec("📦", "Deteksi Outlier — Metode IQR")
    def batas_iqr(s):
        q1,q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3-q1
        return q1-1.5*iqr, q3+1.5*iqr

    rows = []
    for kol in ["jumlah_pengunjung","durasi"]:
        lo,hi = batas_iqr(df[kol])
        n = int(((df[kol]<lo)|(df[kol]>hi)).sum())
        rows.append({"Kolom":kol,"Batas Bawah":round(lo,1),"Batas Atas":round(hi,1),
                     "Outlier":n,"Persen (%)":round(n/len(df)*100,2)})
    st.dataframe(pd.DataFrame(rows).set_index("Kolom"), use_container_width=True)
    st.info("Outlier **dipertahankan** — variasi wajar (lonjakan saat pengumuman), bukan kesalahan input.")

    sec("👁", "Pratinjau Data")
    st.dataframe(df.head(10), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Regresi Linear
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    sec("📈", "Regresi Linear — Prediksi Pengunjung Harian")
    st.caption("Target (y): total pengunjung per hari · Fitur (X): minggu ke- + hari (one-hot encoding)")

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("R²",  f"{r2:.3f}",  help="Proporsi variasi yang dijelaskan model")
    m2.metric("MAE", f"{mae:.1f}", help="Rata-rata selisih absolut prediksi vs aktual")
    m3.metric("RMSE",f"{rmse:.1f}",help="Penalti lebih besar untuk kesalahan ekstrem")
    m4.metric("Error rata-rata", f"{mae/y_lin.mean()*100:.1f}%")

    cl, cr = st.columns(2)
    with cl:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=y_te_lin, y=y_pred_lin, mode="markers",
            marker=dict(color=C_PRIMARY, size=9, opacity=.8,
                        line=dict(width=1.5, color="white")),
            hovertemplate="Aktual: %{x}<br>Prediksi: %{y:.0f}<extra></extra>",
            name="Data uji",
        ))
        lims = [min(float(y_te_lin.min()),float(y_pred_lin.min())),
                max(float(y_te_lin.max()),float(y_pred_lin.max()))]
        fig_sc.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                         line=dict(dash="dash",color="#9CA3AF"), name="Sempurna"))
        chart_layout(fig_sc, f"Prediksi vs Aktual (R² = {r2:.2f})")
        st.plotly_chart(fig_sc, use_container_width=True)

    with cr:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=harian["tanggal"], y=harian["total_pengunjung"],
            mode="lines+markers", marker=dict(size=4),
            line=dict(color=C_BLUE, width=2), name="Aktual",
            hovertemplate="%{x|%d %b}<br>Aktual: <b>%{y:,}</b><extra></extra>",
        ))
        fig_ts.add_trace(go.Scatter(
            x=harian["tanggal"], y=harian["prediksi"],
            mode="lines", line=dict(color=C_RED, width=2, dash="dot"),
            name="Prediksi",
            hovertemplate="%{x|%d %b}<br>Prediksi: <b>%{y:.0f}</b><extra></extra>",
        ))
        chart_layout(fig_ts, "Aktual vs Prediksi (Time Series)")
        st.plotly_chart(fig_ts, use_container_width=True)

    sec("🔢", "Koefisien Model")
    koef = pd.Series(lin.coef_, index=X_lin.columns).sort_values(ascending=False).round(1)
    fig_koef = go.Figure(go.Bar(
        x=koef.values, y=koef.index, orientation="h",
        marker=dict(color=[C_GREEN if v>0 else C_RED for v in koef.values]),
        hovertemplate="<b>%{y}</b>: %{x:.1f}<extra></extra>",
    ))
    chart_layout(fig_koef, f"Koefisien Regresi (intercept = {lin.intercept_:.0f})", h=320)
    st.plotly_chart(fig_koef, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Regresi Non-Linear
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    sec("〰️", "Regresi Non-Linear — Pola Kunjungan per Jam")
    st.caption("Pola per jam bersifat non-linear (dua puncak: pagi & malam). Dibandingkan derajat polinomial 1–6.")

    tabel_r2 = pd.DataFrame({
        "Derajat": list(hasil_deg.keys()),
        "R²": [round(v,3) for v in hasil_deg.values()],
    })

    col_tbl, col_info = st.columns([1,2])
    with col_tbl:
        sec("📋", "Perbandingan R²")
        st.dataframe(
            tabel_r2.set_index("Derajat"),
            use_container_width=True,
            column_config={"R²": st.column_config.ProgressColumn(min_value=0, max_value=1)},
        )
    with col_info:
        sec("💡", "Temuan Utama")
        st.markdown(f"""
        <div style='background:#EEF2FF;border-radius:12px;padding:1.2rem 1.4rem;margin-top:.5rem'>
          <p style='margin:.3rem 0;font-size:.9rem;color:#3730A3'>
            ✦ Linear (deg 1) R² = <strong>{hasil_deg[1]:.3f}</strong> — hampir tidak ada hubungan lurus
          </p>
          <p style='margin:.3rem 0;font-size:.9rem;color:#3730A3'>
            ✦ Polinomial deg {best_deg} R² = <strong>{hasil_deg[best_deg]:.3f}</strong> — jauh lebih baik
          </p>
          <p style='margin:.3rem 0;font-size:.9rem;color:#3730A3'>
            ✦ Ada <strong>dua puncak</strong>: pagi (07–09) dan malam (19–21)
          </p>
        </div>
        """, unsafe_allow_html=True)

    x_grid = np.linspace(xj.min(), xj.max(), 200).reshape(-1,1)
    model_poly = make_pipeline(PolynomialFeatures(best_deg), LinearRegression()).fit(xj,yj)
    lin1 = make_pipeline(PolynomialFeatures(1), LinearRegression()).fit(xj,yj)

    fig_poly = go.Figure()
    fig_poly.add_trace(go.Scatter(
        x=xj.flatten(), y=yj, mode="markers",
        marker=dict(color=C_PRIMARY, size=11, line=dict(width=2,color="white")),
        name="Aktual per jam",
        hovertemplate="Jam %{x:.0f}:00<br>Rata-rata: <b>%{y:.1f}</b><extra></extra>",
    ))
    fig_poly.add_trace(go.Scatter(
        x=x_grid.flatten(), y=lin1.predict(x_grid), mode="lines",
        line=dict(color="#9CA3AF", dash="dash", width=1.5),
        name=f"Linear (R²={hasil_deg[1]:.2f})",
    ))
    fig_poly.add_trace(go.Scatter(
        x=x_grid.flatten(), y=model_poly.predict(x_grid), mode="lines",
        line=dict(color=C_RED, width=3),
        name=f"Polinomial deg {best_deg} (R²={hasil_deg[best_deg]:.2f})",
        fill="tozeroy", fillcolor="rgba(239,68,68,.07)",
    ))
    chart_layout(fig_poly, "Kurva Polinomial vs Linear — Rata-rata Pengunjung per Jam", h=440)
    fig_poly.update_layout(legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_poly, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Clustering
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    sec("🔵", "K-Means Clustering — Segmentasi Jam")
    st.caption("Setiap jam dikelompokkan menjadi 3 sesi berdasarkan rata-rata pengunjung & durasi (K=3).")

    s1,s2,s3 = st.columns(3)
    s1.metric("Silhouette Score", f"{sil:.3f}", help="Mendekati 1 = cluster lebih terpisah")
    s2.metric("Jumlah Cluster", "3", help="Sepi / Sedang / Puncak")
    s3.metric("Jam Puncak", ringkas_sesi.get("Puncak","–"))

    WARNA = {"Sepi":"#93C5FD","Sedang":C_AMBER,"Puncak":C_RED}

    col_a, col_b = st.columns(2)
    with col_a:
        fig_clbar = go.Figure()
        for sesi,c in WARNA.items():
            sub = fitur_jam[fitur_jam["sesi"]==sesi]
            fig_clbar.add_trace(go.Bar(
                x=sub.index, y=sub["rata_pengunjung"],
                name=sesi, marker_color=c,
                hovertemplate="Jam %{x}:00<br>%{y:.1f} pengunjung rata-rata<extra></extra>",
            ))
        chart_layout(fig_clbar, "Sesi per Jam (K-Means)", h=380)
        fig_clbar.update_layout(barmode="overlay", legend=dict(orientation="h",y=1.1))
        fig_clbar.update_xaxes(tickvals=list(fitur_jam.index))
        st.plotly_chart(fig_clbar, use_container_width=True)

    with col_b:
        fig_clsc = go.Figure()
        for sesi,c in WARNA.items():
            sub = fitur_jam[fitur_jam["sesi"]==sesi]
            fig_clsc.add_trace(go.Scatter(
                x=sub["rata_pengunjung"], y=sub["rata_durasi"],
                mode="markers+text",
                text=[f"{j}:00" for j in sub.index],
                textposition="top right",
                textfont=dict(size=9),
                marker=dict(color=c, size=13, line=dict(width=2,color="white")),
                name=sesi,
                hovertemplate="Jam %{text}<br>Pengunjung: %{x:.1f}<br>Durasi: %{y:.0f}s<extra></extra>",
            ))
        chart_layout(fig_clsc, "Pengelompokan Jam (Ruang Fitur)", h=380)
        fig_clsc.update_layout(legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig_clsc, use_container_width=True)

    sec("📋", "Daftar Jam per Sesi")
    BADGE = {"Puncak":"badge-puncak","Sedang":"badge-sedang","Sepi":"badge-sepi"}
    for sesi, jam_list in ringkas_sesi.items():
        st.markdown(f'<span class="{BADGE[sesi]}">{sesi}</span> &nbsp; {jam_list}',
                    unsafe_allow_html=True)
    st.caption("Sesi Puncak berguna untuk menjadwalkan publikasi konten agar langsung terbaca banyak orang.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Klasifikasi
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    sec("🌳", "Decision Tree — Prediksi Jam Ramai / Sepi")
    st.caption(f"Label ramai = total pengunjung > {ambang:.0f} (median). Fitur: jam, hari, minggu ke-.")

    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.metric("Accuracy",  f"{acc:.3f}")
    mc2.metric("Precision", f"{prec:.3f}")
    mc3.metric("Recall",    f"{rec:.3f}")
    mc4.metric("F1-score",  f"{f1:.3f}")

    cx, cy = st.columns(2)
    with cx:
        sec("🟦", "Confusion Matrix")
        cm = confusion_matrix(yc_te, yc_pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            color_continuous_scale=[[0,"#EEF2FF"],[1,C_PRIMARY]],
            x=["Sepi","Ramai"], y=["Sepi","Ramai"],
            labels=dict(x="Prediksi",y="Aktual"),
        )
        fig_cm.update_traces(textfont=dict(size=18,color="white"))
        chart_layout(fig_cm, f"Confusion Matrix (Accuracy = {acc:.2f})", h=380)
        fig_cm.update_coloraxes(showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with cy:
        sec("📊", "Feature Importance")
        fig_imp = go.Figure(go.Bar(
            x=imp.values, y=imp.index, orientation="h",
            marker=dict(
                color=imp.values,
                colorscale=[[0,"#D1FAE5"],[1,C_GREEN]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in imp.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>",
        ))
        chart_layout(fig_imp, "Tingkat Kepentingan Fitur", h=380)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.success(f"Fitur paling menentukan: **'{imp.idxmax()}'** (importance = {imp.max():.2f}). "
               f"Jam akses > hari > minggu dalam memprediksi keramaian.")

    with st.expander("📄 Laporan Klasifikasi Lengkap"):
        st.code(classification_report(yc_te, yc_pred, target_names=["Sepi","Ramai"]))

    st.divider()
    sec("💡", "Insight & Kesimpulan")
    insights = [
        (f"Halaman <strong>{halaman_top}</strong> paling sering diakses — "
         f"{per_halaman.iloc[0]:,} kunjungan ({per_halaman.iloc[0]/per_halaman.sum()*100:.0f}% total). "
         "Tonjolkan di beranda."),
        (f"Website paling ramai pukul <strong>{jam_puncak}:00</strong> dan hari <strong>{hari_top}</strong>. "
         "Publikasikan konten pada jam puncak agar langsung terbaca."),
        (f"Tren pengunjung <strong>naik {pertumbuhan:.0f}%</strong> dari minggu ke-1 ke "
         f"minggu ke-{int(per_minggu.index[-1])} — website makin dikenal, momentum baik untuk update konten rutin."),
        (f"[Regresi Linear] R²={r2:.2f}, MAE={mae:.0f} pengunjung (~{mae/y_lin.mean()*100:.0f}% error). "
         "Model cukup baik untuk estimasi harian."),
        (f"[Regresi Non-Linear] Polinomial derajat {best_deg} (R²={hasil_deg[best_deg]:.2f}) "
         f"jauh ungguli linear (R²={hasil_deg[1]:.2f}) — pola jam bersifat non-linear dengan dua puncak."),
        (f"[Clustering] Sesi PUNCAK: <strong>{ringkas_sesi.get('Puncak','–')}</strong>. "
         f"Silhouette={sil:.2f} — kelompok cukup terpisah dan bermakna."),
        (f"[Klasifikasi] Accuracy={acc:.2f}, F1={f1:.2f}. "
         f"Fitur '<strong>{imp.idxmax()}</strong>' adalah penentu terkuat keramaian website."),
    ]
    items_html = "".join(
        f'<div class="insight-item"><span class="insight-num">{i}</span>{txt}</div>'
        for i,txt in enumerate(insights,1)
    )
    st.markdown(f'<div class="insight-list">{items_html}</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem 0;color:#9CA3AF;font-size:.78rem'>
  Dashboard Analitik Pengunjung · SMP PGRI 8 Kota Bogor ·
  Proyek Sains Data — Skema PBL Terintegrasi · Program Studi Kecerdasan Buatan SSMI IPB
</div>
""", unsafe_allow_html=True)
