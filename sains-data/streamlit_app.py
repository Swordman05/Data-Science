import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.tree import DecisionTreeClassifier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Pengunjung SMP PGRI 8",
    page_icon="🏫",
    layout="wide",
)

BIRU, MERAH, HIJAU, ORANYE = "#2563eb", "#ef4444", "#16a34a", "#f59e0b"
NAMA_HARI = {0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis",
             4: "Jumat", 5: "Sabtu", 6: "Minggu"}

# ── Load & prep data (cached) ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = Path(__file__).parent / "pengunjung_website.csv"
    df = pd.read_csv(csv_path, parse_dates=["tanggal"])
    df["hari"] = df["tanggal"].dt.dayofweek.map(NAMA_HARI)
    df["minggu_ke"] = ((df["tanggal"] - df["tanggal"].min()).dt.days // 7) + 1
    return df

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏫 Dashboard Analitik Pengunjung Website")
st.caption("SMP PGRI 8 Kota Bogor — Proyek Sains Data (Skema PBL Terintegrasi)")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA & Dashboard",
    "🧹 Pembersihan Data",
    "📈 Regresi Linear",
    "〰️ Regresi Non-Linear",
    "🔵 Clustering",
    "🌳 Klasifikasi",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA & Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI
    total_kunjungan = int(df["jumlah_pengunjung"].sum())
    jumlah_hari = df["tanggal"].nunique()
    rata_harian = total_kunjungan / jumlah_hari
    durasi_rata = df["durasi"].mean()

    per_halaman = df.groupby("halaman")["jumlah_pengunjung"].sum().sort_values(ascending=False)
    per_jam = df.groupby("jam")["jumlah_pengunjung"].sum()
    jam_puncak = int(per_jam.idxmax())
    halaman_top = per_halaman.index[0]

    urut_hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    per_hari = df.groupby("hari")["jumlah_pengunjung"].sum().reindex(urut_hari)
    hari_top = per_hari.idxmax()

    per_minggu = df.groupby("minggu_ke")["jumlah_pengunjung"].sum()
    pertumbuhan = (per_minggu.iloc[-1] / per_minggu.iloc[0] - 1) * 100

    st.subheader("KPI Utama")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Kunjungan", f"{total_kunjungan:,}")
    c2.metric("Rata-rata / Hari", f"{rata_harian:,.0f}")
    c3.metric("Halaman Teratas", halaman_top)
    c4.metric("Jam Puncak", f"{jam_puncak}:00")
    c5.metric("Tren Mingguan", f"+{pertumbuhan:.0f}%")

    st.divider()

    # ── Plotly interactive dashboard ─────────────────────────────────────────
    st.subheader("Dashboard Interaktif")
    pengunjung_harian = df.groupby("tanggal")["jumlah_pengunjung"].sum()

    dash = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Pengunjung Harian", "Halaman Terpopuler",
                        "Kunjungan per Jam", "Tren Mingguan"),
    )
    dash.add_trace(go.Scatter(x=pengunjung_harian.index, y=pengunjung_harian.values,
                   mode="lines", fill="tozeroy", line=dict(color=BIRU), name="Harian"),
                   row=1, col=1)
    dash.add_trace(go.Bar(x=per_halaman.values, y=per_halaman.index, orientation="h",
                   marker_color=BIRU, name="Halaman"), row=1, col=2)
    bar_colors = [MERAH if j == jam_puncak else "#93c5fd" for j in per_jam.index]
    dash.add_trace(go.Bar(x=per_jam.index, y=per_jam.values,
                   marker_color=bar_colors, name="Jam"), row=2, col=1)
    dash.add_trace(go.Scatter(x=per_minggu.index, y=per_minggu.values,
                   mode="lines+markers", line=dict(color=HIJAU), name="Mingguan"),
                   row=2, col=2)
    dash.update_yaxes(autorange="reversed", row=1, col=2)
    dash.update_layout(height=600, showlegend=False,
                       margin=dict(t=50, b=20))
    st.plotly_chart(dash, use_container_width=True)

    st.divider()
    st.subheader("EDA — Statistik & Agregasi")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Statistik Ringkas**")
        st.dataframe(df[["jumlah_pengunjung", "durasi"]].describe().round(1))
    with col_b:
        st.markdown("**Kunjungan per Halaman**")
        ringkas_hal = pd.DataFrame({
            "Total": per_halaman,
            "Persen (%)": (per_halaman / per_halaman.sum() * 100).round(1),
        })
        st.dataframe(ringkas_hal)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Kunjungan per Hari**")
        st.dataframe(per_hari.rename("Total").to_frame())
    with col_d:
        st.markdown("**Tren per Minggu**")
        mw_df = pd.DataFrame({
            "Total": per_minggu,
            "Perubahan (%)": per_minggu.pct_change().mul(100).round(1),
        })
        st.dataframe(mw_df)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Pembersihan Data
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Pembersihan Data")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Tipe data**")
        st.dataframe(df.dtypes.rename("Tipe").to_frame())
    with col2:
        st.markdown("**Nilai kosong per kolom**")
        st.dataframe(df.isna().sum().rename("Jumlah kosong").to_frame())

    n_dup = int(df.duplicated().sum())
    st.info(f"Duplikat: **{n_dup}** baris {'— dihapus' if n_dup > 0 else '(tidak ada)'}")

    st.markdown("**Validasi rentang nilai**")
    masalah = {
        "jam di luar 0-23":       int((~df["jam"].between(0, 23)).sum()),
        "jumlah_pengunjung <= 0": int((df["jumlah_pengunjung"] <= 0).sum()),
        "durasi <= 0 detik":      int((df["durasi"] <= 0).sum()),
    }
    val_df = pd.DataFrame([
        {"Masalah": k, "Pelanggaran": v,
         "Status": "✅ OK" if v == 0 else "⚠️ Perlu dibersihkan"}
        for k, v in masalah.items()
    ]).set_index("Masalah")
    st.dataframe(val_df)

    st.markdown("**Deteksi Outlier (metode IQR)**")
    def batas_iqr(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    rows = []
    for kol in ["jumlah_pengunjung", "durasi"]:
        low, high = batas_iqr(df[kol])
        n_out = int(((df[kol] < low) | (df[kol] > high)).sum())
        rows.append({"Kolom": kol, "Batas Bawah": round(low, 1),
                     "Batas Atas": round(high, 1),
                     "Jumlah Outlier": n_out,
                     "Persen (%)": round(n_out / len(df) * 100, 2)})
    st.dataframe(pd.DataFrame(rows).set_index("Kolom"))
    st.success("Outlier **dipertahankan** — merupakan variasi wajar (mis. lonjakan saat pengumuman), bukan kesalahan input.")

    st.markdown("**Pratinjau data (5 baris pertama)**")
    st.dataframe(df.head())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Regresi Linear
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Regresi Linear — Prediksi Pengunjung Harian")
    st.markdown("**Target (y):** total pengunjung per hari · **Fitur (X):** minggu ke- + hari (one-hot)")

    @st.cache_data
    def run_linear(df):
        harian = (df.groupby("tanggal")
                    .agg(total_pengunjung=("jumlah_pengunjung", "sum"))
                    .reset_index())
        harian["minggu_ke"] = ((harian["tanggal"] - harian["tanggal"].min()).dt.days // 7) + 1
        harian["nama_hari"] = harian["tanggal"].dt.dayofweek.map(NAMA_HARI)
        dummies = pd.get_dummies(harian["nama_hari"], prefix="hari", drop_first=True)
        X = pd.concat([harian[["minggu_ke"]], dummies], axis=1).astype(float)
        y = harian["total_pengunjung"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        lin = LinearRegression().fit(X_tr, y_tr)
        y_pred = lin.predict(X_te)
        harian["prediksi"] = lin.predict(X)
        return harian, lin, X, y, X_tr, X_te, y_tr, y_te, y_pred

    harian, lin, X, y, X_tr, X_te, y_tr, y_te, y_pred = run_linear(df)

    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    rmse = root_mean_squared_error(y_te, y_pred)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", f"{r2:.3f}")
    m2.metric("MAE", f"{mae:.1f} pengunjung")
    m3.metric("RMSE", f"{rmse:.1f} pengunjung")
    m4.metric("Error rata-rata", f"{mae/y.mean()*100:.1f}%")

    col_l, col_r = st.columns(2)
    with col_l:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=y_te, y=y_pred, mode="markers",
                         marker=dict(color=BIRU, size=8, opacity=0.8), name="Prediksi"))
        lims = [min(float(y_te.min()), float(y_pred.min())),
                max(float(y_te.max()), float(y_pred.max()))]
        fig_sc.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                         line=dict(dash="dash", color="gray"), name="Sempurna"))
        fig_sc.update_layout(title=f"Prediksi vs Aktual (R²={r2:.2f})",
                             xaxis_title="Aktual", yaxis_title="Prediksi", height=380)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_r:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=harian["tanggal"], y=harian["total_pengunjung"],
                         mode="lines+markers", marker=dict(size=4),
                         line=dict(color=BIRU), name="Aktual"))
        fig_ts.add_trace(go.Scatter(x=harian["tanggal"], y=harian["prediksi"],
                         mode="lines", line=dict(color=MERAH, width=2), name="Prediksi"))
        fig_ts.update_layout(title="Aktual vs Prediksi (time series)",
                             xaxis_title="Tanggal", yaxis_title="Pengunjung", height=380)
        st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("**Koefisien model** (basis: Jumat)")
    koef = pd.Series(lin.coef_, index=X.columns).sort_values(ascending=False).round(1)
    st.dataframe(koef.rename("Koefisien").to_frame())
    st.caption(f"Intercept: {lin.intercept_:.0f} pengunjung")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Regresi Non-Linear
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Regresi Non-Linear — Pola Kunjungan per Jam")
    st.markdown("Pola kunjungan per jam **tidak linear** (ada dua puncak). Dibandingkan beberapa derajat polinomial.")

    @st.cache_data
    def run_poly(df):
        per_jam_rata = (df.groupby(["tanggal", "jam"])["jumlah_pengunjung"].sum()
                          .groupby("jam").mean())
        xj = per_jam_rata.index.values.reshape(-1, 1).astype(float)
        yj = per_jam_rata.values
        hasil_deg = {}
        for d in [1, 2, 3, 4, 6]:
            mdl = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(xj, yj)
            hasil_deg[d] = r2_score(yj, mdl.predict(xj))
        best_deg = max(hasil_deg, key=hasil_deg.get)
        return xj, yj, hasil_deg, best_deg

    xj, yj, hasil_deg, best_deg = run_poly(df)

    st.markdown("**Perbandingan R² per derajat polinomial**")
    tabel = pd.DataFrame({
        "Derajat": list(hasil_deg.keys()),
        "R²": [round(v, 3) for v in hasil_deg.values()],
    })
    st.dataframe(tabel.set_index("Derajat"))
    st.info(f"Derajat terbaik: **{best_deg}** (R² = {hasil_deg[best_deg]:.3f}) vs linear R² = {hasil_deg[1]:.3f}")

    x_grid = np.linspace(xj.min(), xj.max(), 200).reshape(-1, 1)
    model_poly = make_pipeline(PolynomialFeatures(best_deg), LinearRegression()).fit(xj, yj)
    lin1 = make_pipeline(PolynomialFeatures(1), LinearRegression()).fit(xj, yj)

    fig_poly = go.Figure()
    fig_poly.add_trace(go.Scatter(x=xj.flatten(), y=yj, mode="markers",
                       marker=dict(color=BIRU, size=9), name="Aktual per jam"))
    fig_poly.add_trace(go.Scatter(x=x_grid.flatten(), y=lin1.predict(x_grid),
                       mode="lines", line=dict(color="gray", dash="dash"),
                       name=f"Linear (R²={hasil_deg[1]:.2f})"))
    fig_poly.add_trace(go.Scatter(x=x_grid.flatten(), y=model_poly.predict(x_grid),
                       mode="lines", line=dict(color=MERAH, width=2.5),
                       name=f"Polinomial deg {best_deg} (R²={hasil_deg[best_deg]:.2f})"))
    fig_poly.update_layout(title="Regresi Non-Linear: Rata-rata Pengunjung per Jam",
                           xaxis_title="Jam (24h)", yaxis_title="Rata-rata Pengunjung",
                           height=420)
    st.plotly_chart(fig_poly, use_container_width=True)
    st.caption("Kurva polinomial menangkap dua puncak (pagi & malam) yang tidak bisa dijelaskan oleh garis lurus.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Clustering K-Means
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("K-Means Clustering — Segmentasi Jam menjadi Sesi")
    st.markdown("Fitur per jam: **rata-rata pengunjung** & **rata-rata durasi**. K=3 sesi: Sepi / Sedang / Puncak.")

    @st.cache_data
    def run_kmeans(df):
        fitur_jam = pd.DataFrame({
            "rata_pengunjung": df.groupby(["tanggal", "jam"])["jumlah_pengunjung"].sum()
                                  .groupby("jam").mean(),
            "rata_durasi": df.groupby("jam")["durasi"].mean(),
        })
        X_clu = StandardScaler().fit_transform(fitur_jam)
        km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_clu)
        fitur_jam["cluster"] = km.labels_
        urut = (fitur_jam.groupby("cluster")["rata_pengunjung"].mean()
                .sort_values().index.tolist())
        label_sesi = {urut[0]: "Sepi", urut[1]: "Sedang", urut[2]: "Puncak"}
        fitur_jam["sesi"] = fitur_jam["cluster"].map(label_sesi)
        sil = silhouette_score(X_clu, km.labels_)
        return fitur_jam, sil

    from sklearn.metrics import silhouette_score
    fitur_jam, sil = run_kmeans(df)

    WARNA_SESI = {"Sepi": "#93c5fd", "Sedang": ORANYE, "Puncak": MERAH}

    st.metric("Silhouette Score (k=3)", f"{sil:.3f}", help="Mendekati 1 = cluster lebih terpisah")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_bar = go.Figure()
        for sesi, c in WARNA_SESI.items():
            sub = fitur_jam[fitur_jam["sesi"] == sesi]
            fig_bar.add_trace(go.Bar(x=sub.index, y=sub["rata_pengunjung"],
                              name=sesi, marker_color=c))
        fig_bar.update_layout(title="Sesi Tiap Jam (K-Means)", barmode="stack",
                              xaxis_title="Jam", yaxis_title="Rata-rata Pengunjung",
                              height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        fig_sc2 = go.Figure()
        for sesi, c in WARNA_SESI.items():
            sub = fitur_jam[fitur_jam["sesi"] == sesi]
            fig_sc2.add_trace(go.Scatter(
                x=sub["rata_pengunjung"], y=sub["rata_durasi"],
                mode="markers+text",
                text=[f"{j}:00" for j in sub.index],
                textposition="top right",
                marker=dict(color=c, size=10),
                name=sesi))
        fig_sc2.update_layout(title="Pengelompokan Jam (ruang fitur)",
                              xaxis_title="Rata-rata Pengunjung",
                              yaxis_title="Rata-rata Durasi (detik)", height=380)
        st.plotly_chart(fig_sc2, use_container_width=True)

    st.markdown("**Daftar jam per sesi**")
    ringkas_sesi = (fitur_jam.reset_index()
                    .groupby("sesi")["jam"]
                    .agg(lambda s: ", ".join(f"{j}:00" for j in sorted(s)))
                    .reindex(["Puncak", "Sedang", "Sepi"]))
    for sesi, jam_list in ringkas_sesi.items():
        st.markdown(f"- **{sesi}**: {jam_list}")

    st.caption("Sesi Puncak = jam pagi (07-09) & malam (19-21). Berguna untuk menjadwalkan publikasi konten.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Klasifikasi
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Decision Tree Klasifikasi — Prediksi Jam Ramai / Sepi")
    st.markdown("Label **ramai = 1** bila total pengunjung per (tanggal, jam) di atas median. Fitur: jam, hari, minggu ke-.")

    @st.cache_data
    def run_clf(df):
        tj = (df.groupby(["tanggal", "jam"])["jumlah_pengunjung"].sum()
                .reset_index(name="total"))
        tj["dayofweek"] = tj["tanggal"].dt.dayofweek
        tj["minggu_ke"] = ((tj["tanggal"] - tj["tanggal"].min()).dt.days // 7) + 1
        ambang = tj["total"].median()
        tj["ramai"] = (tj["total"] > ambang).astype(int)
        dum = pd.get_dummies(tj["dayofweek"].map(NAMA_HARI), prefix="hari")
        Xc = pd.concat([tj[["jam", "minggu_ke"]], dum], axis=1).astype(float)
        yc = tj["ramai"]
        Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
            Xc, yc, test_size=0.25, random_state=42, stratify=yc)
        clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xc_tr, yc_tr)
        yc_pred = clf.predict(Xc_te)
        return clf, Xc, yc_tr, yc_te, yc_pred, ambang

    clf, Xc, yc_tr, yc_te, yc_pred, ambang = run_clf(df)

    acc   = accuracy_score(yc_te, yc_pred)
    prec  = precision_score(yc_te, yc_pred)
    rec   = recall_score(yc_te, yc_pred)
    f1    = f1_score(yc_te, yc_pred)

    st.caption(f"Ambang 'ramai' (median total per jam): **{ambang:.0f} pengunjung**")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy",  f"{acc:.3f}")
    mc2.metric("Precision", f"{prec:.3f}")
    mc3.metric("Recall",    f"{rec:.3f}")
    mc4.metric("F1-score",  f"{f1:.3f}")

    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(yc_te, yc_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           x=["Sepi", "Ramai"], y=["Sepi", "Ramai"],
                           labels=dict(x="Prediksi", y="Aktual"),
                           title=f"Confusion Matrix (Accuracy={acc:.2f})")
        fig_cm.update_layout(height=360)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_y:
        st.markdown("**Tingkat Kepentingan Fitur**")
        imp = pd.Series(clf.feature_importances_, index=Xc.columns)
        imp = imp[imp > 0].sort_values()
        fig_imp = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                            marker_color=HIJAU))
        fig_imp.update_layout(title="Feature Importance", xaxis_title="Importance",
                              height=360)
        st.plotly_chart(fig_imp, use_container_width=True)

    fitur_utama = imp.idxmax()
    st.success(f"Fitur paling menentukan: **'{fitur_utama}'** (importance = {imp.max():.2f})")

    with st.expander("Laporan Klasifikasi Lengkap"):
        st.text(classification_report(yc_te, yc_pred, target_names=["Sepi", "Ramai"]))

    st.divider()
    st.subheader("📝 Insight & Kesimpulan")
    per_halaman_ins = df.groupby("halaman")["jumlah_pengunjung"].sum().sort_values(ascending=False)
    per_jam_ins = df.groupby("jam")["jumlah_pengunjung"].sum()
    per_minggu_ins = df.groupby("minggu_ke")["jumlah_pengunjung"].sum()
    tumbuh = (per_minggu_ins.iloc[-1] / per_minggu_ins.iloc[0] - 1) * 100

    fitur_jam_ins, sil_ins = run_kmeans(df)
    ringkas_puncak = (fitur_jam_ins.reset_index()
                      .groupby("sesi")["jam"]
                      .agg(lambda s: ", ".join(f"{j}:00" for j in sorted(s)))
                      .get("Puncak", "-"))

    _, lin_ins, X_ins, y_ins, _, _, _, y_te_ins, y_pred_ins = run_linear(df)
    r2_ins = r2_score(y_te_ins, y_pred_ins)
    mae_ins = mean_absolute_error(y_te_ins, y_pred_ins)
    _, _, hasil_deg_ins, best_deg_ins = run_poly(df)

    insights = [
        f"Halaman **'{per_halaman_ins.index[0]}'** paling sering diakses "
        f"({per_halaman_ins.iloc[0]:,} kunjungan / "
        f"{per_halaman_ins.iloc[0]/per_halaman_ins.sum()*100:.0f}% total).",
        f"Website paling ramai pukul **{int(per_jam_ins.idxmax())}:00**; "
        f"hari paling ramai **{df.groupby('hari')['jumlah_pengunjung'].sum().reindex(['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']).idxmax()}**.",
        f"Tren pengunjung **naik {tumbuh:.0f}%** dari minggu ke-1 ke minggu ke-{int(per_minggu_ins.index[-1])}.",
        f"[Regresi Linear] R²={r2_ins:.2f}, MAE={mae_ins:.0f} (~{mae_ins/y_ins.mean()*100:.0f}% dari rata-rata).",
        f"[Regresi Non-Linear] Polinomial derajat {best_deg_ins} (R²={hasil_deg_ins[best_deg_ins]:.2f}) "
        f"jauh ungguli linear (R²={hasil_deg_ins[1]:.2f}).",
        f"[Clustering] Sesi PUNCAK = {ringkas_puncak} (silhouette={sil_ins:.2f}).",
        f"[Klasifikasi] Accuracy={acc:.2f}, F1={f1:.2f}; fitur penentu = **'{fitur_utama}'**.",
    ]
    for i, ins in enumerate(insights, 1):
        st.markdown(f"{i}. {ins}")
