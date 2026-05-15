# =============================================================================
# OLAH DATA BAB 4.1 — PKM RE 2026 (VERSI LOCAL PYTHON)
# Efisiensi Komputasi dan Kecepatan Inferensi
#
# CARA PAKAI:
#   1. Pastikan struktur folder:
#        project/
#          ├── data/summary_inference_metrics_llm_inference_oracle_top15_full50_v1.csv
#          └── bab4_1_inference.py  (file ini)
#   2. Install dependensi (sekali saja):
#        pip install -r requirements.txt
#   3. Jalankan:
#        python bab4_1_inference.py
# =============================================================================

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# KONFIGURASI — sesuaikan path jika struktur folder Anda berbeda
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
CSV_PATH = SCRIPT_DIR / "data" / "summary_inference_metrics_llm_inference_oracle_top15_full50_v1.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# VALIDASI INPUT
# =============================================================================

if not CSV_PATH.exists():
    print(f"❌ ERROR: File CSV tidak ditemukan di: {CSV_PATH}")
    print(f"\n   Pastikan file CSV sudah ada di folder 'data/' di lokasi:")
    print(f"   {CSV_PATH}")
    print(f"\n   Atau ubah variabel CSV_PATH di baris 25 script ini.")
    sys.exit(1)

print(f"📂 Membaca CSV dari: {CSV_PATH}")
print(f"💾 Output akan disimpan di: {OUTPUT_DIR}\n")


# =============================================================================
# LOAD & SIAPKAN DATA
# =============================================================================

df = pd.read_csv(CSV_PATH)

# Bersihkan nama model: "Pruned_10P_Aligned" -> "Pruned_10P"
df['model_short'] = df['model_name'].str.replace('_Aligned', '', regex=False)

# Urutkan model sesuai urutan logis penelitian
order = ['Base_14B', 'Pruned_10P', 'Pruned_20P', 'Pruned_30P',
         'Pruned_10P_FT', 'Pruned_20P_FT', 'Pruned_30P_FT']
df['model_short'] = pd.Categorical(df['model_short'], categories=order, ordered=True)
df = df.sort_values('model_short').reset_index(drop=True)

# Pilih kolom utama untuk 4 panel
df_eff = df[['model_short',
             'avg_ttft',
             'avg_end_to_end_latency',
             'avg_peak_vram_total_gb',
             'avg_throughput_tokens_per_sec']].copy()
df_eff.columns = ['Model', 'TTFT_s', 'Latency_s', 'VRAM_GB', 'Throughput_tps']

print("=" * 75)
print("Tabel Efisiensi Inferensi — n=50 queries")
print("=" * 75)
print(df_eff.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# =============================================================================
# GAMBAR — BAR CHART EFISIENSI INFERENSI (4 PANEL)
# TTFT, Total Latency, Peak VRAM, Throughput
# =============================================================================

plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})

# Palet warna konsisten 7 model
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2']

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Perbandingan Metrik Inferensi — 7 Varian Model (Two-Stage Generative)',
             fontsize=14, fontweight='bold', y=1.02)


def ylim_from_zero(values, pad_hi=0.18):
    """Y-axis mulai dari 0 supaya semua bar terlihat penuh."""
    vmax = float(np.max(values))
    if vmax <= 0:
        vmax = 1.0
    return (0, vmax * (1 + pad_hi))


metrics_inf = [
    ('TTFT_s',         'TTFT (s)',           '{:.4f}'),
    ('Latency_s',      'Total Latency (s)',  '{:.2f}'),
    ('VRAM_GB',        'Peak VRAM (GB)',     '{:.2f}'),
    ('Throughput_tps', 'Throughput (tok/s)', '{:.1f}'),
]

for ax, (col, title, fmt) in zip(axes, metrics_inf):
    values = df_eff[col].values
    bars = ax.bar(df_eff['Model'].astype(str), values, color=colors,
                  edgecolor='black', linewidth=0.6)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(title)
    ax.set_ylim(ylim_from_zero(values))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    # Anotasi nilai di atas bar dengan offset kecil supaya tidak menempel
    vmax = float(np.max(values))
    offset = vmax * 0.015
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                fmt.format(v), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
out_gambar = OUTPUT_DIR / 'gambar_efisiensi_inferensi.png'
plt.savefig(out_gambar, dpi=300, bbox_inches='tight')
plt.show()
print(f"\n✓ Tersimpan: {out_gambar}")


# =============================================================================
# RINGKASAN PERUBAHAN RELATIF TERHADAP BASELINE (untuk narasi Bab 4.1)
# =============================================================================

base_row = df_eff[df_eff['Model'].astype(str) == 'Base_14B'].iloc[0]
print("\n" + "=" * 75)
print("BASELINE (Base_14B):")
print(f"  TTFT       = {base_row['TTFT_s']:.4f} s")
print(f"  Latency    = {base_row['Latency_s']:.4f} s")
print(f"  Peak VRAM  = {base_row['VRAM_GB']:.4f} GB")
print(f"  Throughput = {base_row['Throughput_tps']:.4f} tok/s")
print("=" * 75)
print("PERUBAHAN RELATIF TERHADAP Base_14B:")
print("-" * 75)
for _, r in df_eff.iterrows():
    m = str(r['Model'])
    if m == 'Base_14B':
        continue
    d_ttft = (r['TTFT_s'] - base_row['TTFT_s']) / base_row['TTFT_s'] * 100
    d_lat  = (r['Latency_s'] - base_row['Latency_s']) / base_row['Latency_s'] * 100
    d_vram = (r['VRAM_GB'] - base_row['VRAM_GB']) / base_row['VRAM_GB'] * 100
    d_tp   = (r['Throughput_tps'] - base_row['Throughput_tps']) / base_row['Throughput_tps'] * 100
    print(f"  • {m:18s} | ΔTTFT={d_ttft:+7.2f}% | ΔLat={d_lat:+7.2f}% | "
          f"ΔVRAM={d_vram:+6.2f}% | ΔTP={d_tp:+6.2f}%")


# =============================================================================
# EXPORT TABEL UNTUK LAPORAN
# =============================================================================

out_tabel = OUTPUT_DIR / 'tabel_efisiensi_inferensi.csv'
df_eff.to_csv(out_tabel, index=False, float_format='%.4f')
print(f"\n✓ Tersimpan: {out_tabel}")

print("\n" + "=" * 75)
print(f"✅ SELESAI! Semua output ada di folder: {OUTPUT_DIR}")
print("=" * 75)