"""
generate_data.py
=================
Membuat dataset contoh "Data Pengunjung Website Sekolah" (SMP PGRI 8).

Catatan: Website sekolah ini baru, jadi kita belum punya log traffic asli.
Script ini membuat data sintetis yang REALISTIS dan REPRODUCIBLE (pakai seed
tetap) agar analisis di notebook bisa dijalankan ulang oleh siapa pun dengan
hasil yang sama.

Kolom yang dihasilkan (mengembangkan contoh tugas dengan tambahan kolom `jam`
supaya analisis "jam paling ramai" bisa dilakukan):

    tanggal            : tanggal kunjungan (YYYY-MM-DD)
    jam                : jam kunjungan (0-23)
    halaman            : halaman website yang diakses
    jumlah_pengunjung  : jumlah kunjungan ke halaman itu pada jam tsb
    durasi             : rata-rata durasi kunjungan (detik)

Jalankan:  python generate_data.py
Output  :  pengunjung_website.csv
"""

import numpy as np
import pandas as pd

# Seed tetap -> data selalu sama setiap kali di-generate (reproducible)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Parameter dasar
# ---------------------------------------------------------------------------

# Rentang 8 minggu penuh (Senin-Minggu) supaya tren mingguan rapi
TANGGAL = pd.date_range("2026-04-06", "2026-05-31", freq="D")  # 56 hari

# Jam aktif website (orang jarang buka dini hari)
JAM = list(range(6, 24))  # 06:00 - 23:00

# Halaman website + bobot popularitas dasar + rata-rata durasi (detik).
# /pengumuman sengaja dibuat PALING populer (sesuai insight tugas),
# tapi /ppdb & /berita punya durasi lebih lama (engagement lebih tinggi).
HALAMAN = {
    "/pengumuman":  {"bobot": 1.00, "durasi": 95},
    "/":            {"bobot": 0.85, "durasi": 55},
    "/berita":      {"bobot": 0.80, "durasi": 185},
    "/ppdb":        {"bobot": 0.55, "durasi": 210},
    "/galeri":      {"bobot": 0.45, "durasi": 120},
    "/fasilitas":   {"bobot": 0.35, "durasi": 110},
    "/profil":      {"bobot": 0.30, "durasi": 150},
    "/guru-staff":  {"bobot": 0.25, "durasi": 80},
    "/kontak":      {"bobot": 0.15, "durasi": 45},
}

# ---------------------------------------------------------------------------
# 2. Pola perilaku pengunjung (membuat data terasa nyata)
# ---------------------------------------------------------------------------

# Pola per JAM: ada 2 puncak -> pagi (07-09, sebelum/saat sekolah) dan
# malam (19-21, orang tua cek setelah pulang kerja). Puncak terbesar 20:00.
def bobot_jam(jam: int) -> float:
    pagi = np.exp(-((jam - 8) ** 2) / 4.0)     # puncak ~08:00
    malam = 1.25 * np.exp(-((jam - 20) ** 2) / 3.0)  # puncak ~20:00 (lebih tinggi)
    return 0.15 + pagi + malam

# Pola per HARI: hari kerja lebih ramai dari akhir pekan. Senin tertinggi
# (pengumuman baru biasanya diposting awal pekan).
BOBOT_HARI = {0: 1.20, 1: 1.05, 2: 1.00, 3: 1.00, 4: 0.95, 5: 0.70, 6: 0.65}
#             Sen     Sel     Rab     Kam     Jum     Sab     Min

# ---------------------------------------------------------------------------
# 3. Bangun dataset baris per (tanggal, jam, halaman)
# ---------------------------------------------------------------------------

baris = []
minggu_awal = TANGGAL.min()

for tgl in TANGGAL:
    minggu_ke = (tgl - minggu_awal).days // 7  # 0..7
    # Traffic keseluruhan naik perlahan tiap minggu (website makin dikenal)
    faktor_minggu = 1.0 + 0.06 * minggu_ke
    # PPDB makin ramai di minggu-minggu akhir (musim pendaftaran)
    faktor_ppdb = 1.0 + 0.22 * minggu_ke

    w_hari = BOBOT_HARI[tgl.dayofweek]

    for jam in JAM:
        w_jam = bobot_jam(jam)
        for hal, cfg in HALAMAN.items():
            faktor = faktor_ppdb if hal == "/ppdb" else faktor_minggu
            rata = cfg["bobot"] * w_jam * w_hari * faktor * 11.0
            # Jumlah pengunjung ~ Poisson (cocok untuk data hitungan/count)
            jumlah = int(rng.poisson(max(rata, 0.05)))
            if jumlah == 0:
                continue  # tidak ada kunjungan -> tidak dicatat
            # Durasi rata-rata + sedikit variasi acak
            durasi = int(max(10, rng.normal(cfg["durasi"], cfg["durasi"] * 0.18)))
            baris.append((tgl.date().isoformat(), jam, hal, jumlah, durasi))

df = pd.DataFrame(baris, columns=["tanggal", "jam", "halaman", "jumlah_pengunjung", "durasi"])

# Urutkan biar rapi
df = df.sort_values(["tanggal", "jam", "halaman"]).reset_index(drop=True)

df.to_csv("pengunjung_website.csv", index=False)

# Ringkasan singkat saat dijalankan
print(f"Dataset dibuat: pengunjung_website.csv")
print(f"Jumlah baris   : {len(df):,}")
print(f"Rentang tanggal: {df['tanggal'].min()} s/d {df['tanggal'].max()}")
print(f"Total kunjungan: {df['jumlah_pengunjung'].sum():,}")
print("\nHalaman paling sering dikunjungi (total):")
print(df.groupby("halaman")["jumlah_pengunjung"].sum().sort_values(ascending=False).head())
