# =============================================================================
# OLAH DATA BAB 4 — PKM RE 2026 (VERSI LOCAL PYTHON)
# Bab 4.2: Kualitas Respon (Repaired Metrics) + Pareto Latency vs NDCG
#
# CARA PAKAI:
#   1. Pastikan struktur folder:
#        project/
#          ├── data/summary_quality_metrics_llm_only_oracle_ndcg_full50_v2_top15.csv
#          └── bab4_kualitas_pareto.py  (file ini)
#   2. Install dependensi (sekali saja):
#        pip install -r requirements.txt
#   3. Jalankan:
#        python bab4_kualitas_pareto.py
# =============================================================================

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# =============================================================================
# KONFIGURASI — sesuaikan path jika struktur folder Anda berbeda
# =============================================================================

# Cari folder script ini berada
SCRIPT_DIR = Path(__file__).parent.resolve()

# Path file CSV input
CSV_PATH = SCRIPT_DIR / "data" / "summary_quality_metrics_llm_only_oracle_ndcg_full50_v2_top15.csv"

# Folder untuk menyimpan output (otomatis dibuat jika belum ada)
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# VALIDASI INPUT — pastikan file CSV ada sebelum diproses
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
df['model_short'] = df['model_name'].str.replace('_Aligned', '', regex=False)

order = ['Base_14B', 'Pruned_10P', 'Pruned_20P', 'Pruned_30P',
         'Pruned_10P_FT', 'Pruned_20P_FT', 'Pruned_30P_FT']
df['model_short'] = pd.Categorical(df['model_short'], categories=order, ordered=True)
df = df.sort_values('model_short').reset_index(drop=True)

df_q = df[['model_short',
           'avg_repaired_llm_ndcg@10_attainable',
           'avg_repaired_llm_recall@10_attainable',
           'avg_repaired_llm_hit_rate@10',
           'avg_repaired_llm_precision@10',
           'parse_success_rate_strict_top10']].copy()
df_q.columns = ['Model', 'NDCG@10', 'Recall@10', 'HitRate@10', 'Precision@10', 'ParseSuccess']

print("=" * 70)
print("Tabel Kualitas Respon (Repaired Metrics) — n=50 queries")
print("=" * 70)
print(df_q.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# =============================================================================
# DATA LATENCY DAN VRAM (dari Tabel 1 laporan)
# =============================================================================

latency_data = {
    'Base_14B':       6.42, 'Pruned_10P':     6.47, 'Pruned_20P':     6.32,
    'Pruned_30P':     6.20, 'Pruned_10P_FT':  3.42, 'Pruned_20P_FT':  3.63,
    'Pruned_30P_FT':  5.54,
}
vram_data = {
    'Base_14B':       30.44, 'Pruned_10P':     28.36, 'Pruned_20P':     26.28,
    'Pruned_30P':     24.35, 'Pruned_10P_FT':  28.36, 'Pruned_20P_FT':  26.28,
    'Pruned_30P_FT':  24.35,
}
df_q['Latency_s'] = df_q['Model'].astype(str).map(latency_data)
df_q['VRAM_GB']   = df_q['Model'].astype(str).map(vram_data)


# =============================================================================
# GAMBAR 1 — BAR CHART METRIK KUALITAS RESPON (4 PANEL)
# =============================================================================

plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2']

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Perbandingan Metrik Kualitas Respon — 7 Varian Model (Two-Stage Generative)',
             fontsize=14, fontweight='bold', y=1.02)

metrics_q = [
    ('NDCG@10',     'NDCG@10 (Attainable)',  (0.75, 0.86)),
    ('Recall@10',   'Recall@10 (Attainable)',(0.94, 1.00)),
    ('HitRate@10',  'Hit Rate@10',           (0.0,  1.05)),
    ('Precision@10','Precision@10',          (0.74, 0.81)),
]

for ax, (col, title, ylim) in zip(axes, metrics_q):
    bars = ax.bar(df_q['Model'].astype(str), df_q[col], color=colors,
                  edgecolor='black', linewidth=0.6)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(title)
    ax.set_ylim(ylim)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    for bar, v in zip(bars, df_q[col]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
out_gambar1 = OUTPUT_DIR / 'gambar_kualitas_respon.png'
plt.savefig(out_gambar1, dpi=300, bbox_inches='tight')
plt.show()  # Akan buka window matplotlib di local
print(f"\n✓ Tersimpan: {out_gambar1}")


# =============================================================================
# GAMBAR 2 — KURVA PARETO STATIS (matplotlib): NDCG@10 vs LATENCY
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 7))

x = df_q['Latency_s'].values
y = df_q['NDCG@10'].values
labels = df_q['Model'].astype(str).values

for i in range(len(df_q)):
    ax.scatter(x[i], y[i], s=240, color=colors[i],
               edgecolor='black', linewidth=1.2, zorder=3, label=labels[i])


def pareto_front(xs, ys):
    """Hitung Pareto frontier: minimisasi X, maksimisasi Y."""
    pts = sorted(zip(xs, ys, range(len(xs))), key=lambda p: (p[0], -p[1]))
    front = []
    best_y = -np.inf
    for px, py, pi in pts:
        if py > best_y:
            front.append((px, py, pi))
            best_y = py
    return front


front = pareto_front(x, y)
fx = [p[0] for p in front]
fy = [p[1] for p in front]
front_idx = [p[2] for p in front]

ax.plot(fx, fy, '--', color='red', linewidth=2, alpha=0.7,
        label='Pareto Frontier', zorder=2)

for idx in front_idx:
    ax.scatter(x[idx], y[idx], s=420, facecolors='none',
               edgecolors='red', linewidth=2.2, zorder=4)

offsets = {
    'Base_14B':       (-10, 8),  'Pruned_10P':     (-10, -16),
    'Pruned_20P':     (10,  10), 'Pruned_30P':     (-10, -16),
    'Pruned_10P_FT':  (10, 8),   'Pruned_20P_FT':  (10, 8),
    'Pruned_30P_FT':  (10, -16),
}
for i, lbl in enumerate(labels):
    dx, dy = offsets.get(lbl, (10, 8))
    ha = 'right' if dx < 0 else 'left'
    ax.annotate(lbl, (x[i], y[i]), xytext=(dx, dy),
                textcoords='offset points', fontsize=10, fontweight='bold', ha=ha)

ax.annotate('', xy=(3.2, 0.853), xytext=(4.5, 0.842),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(4.55, 0.840, 'Arah Ideal\n(cepat & akurat)',
        fontsize=9, color='green', fontweight='bold', ha='left')

ax.set_xlabel('Total Latency (detik) — kanan = lebih cepat, lebih baik',
              fontsize=12, fontweight='bold')
ax.set_ylabel('NDCG@10 Repaired (Attainable) — atas = lebih akurat',
              fontsize=12, fontweight='bold')
ax.set_title('Kurva Pareto Trade-off: Kualitas Rekomendasi vs Latensi Inferensi',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(6.8, 3.0)
ax.set_ylim(0.785, 0.855)
ax.legend(loc='lower left', fontsize=9, framealpha=0.95, ncol=2)

plt.tight_layout()
out_gambar2 = OUTPUT_DIR / 'gambar_pareto_ndcg_vs_latency.png'
plt.savefig(out_gambar2, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Tersimpan: {out_gambar2}")


# =============================================================================
# RINGKASAN PARETO-OPTIMAL UNTUK NARASI BAB 4
# =============================================================================

print("\n" + "=" * 70)
print("MODEL PADA PARETO FRONTIER (Latency vs NDCG):")
print("=" * 70)
for px, py, pi in front:
    print(f"  • {labels[pi]:18s} | Latency = {px:.2f} s | NDCG@10 = {py:.4f}")

base_ndcg = df_q.loc[df_q['Model'].astype(str) == 'Base_14B', 'NDCG@10'].values[0]
base_lat  = df_q.loc[df_q['Model'].astype(str) == 'Base_14B', 'Latency_s'].values[0]

print("\n" + "=" * 70)
print(f"PERUBAHAN RELATIF TERHADAP Base_14B (NDCG@10 = {base_ndcg:.4f}, Latency = {base_lat:.2f} s):")
print("=" * 70)
for i, m in enumerate(labels):
    if m == 'Base_14B':
        continue
    d_ndcg = (df_q['NDCG@10'].iloc[i] - base_ndcg) / base_ndcg * 100
    d_lat  = (df_q['Latency_s'].iloc[i] - base_lat) / base_lat * 100
    print(f"  • {m:18s} | ΔNDCG = {d_ndcg:+6.2f}% | ΔLatency = {d_lat:+6.2f}%")


# =============================================================================
# EXPORT TABEL UNTUK LAPORAN (CSV)
# =============================================================================

df_export = df_q[['Model', 'NDCG@10', 'Recall@10', 'HitRate@10',
                  'Precision@10', 'ParseSuccess', 'Latency_s']].copy()
df_export.columns = ['Model', 'NDCG@10', 'Recall@10', 'Hit Rate@10',
                     'Precision@10', 'Parse Success', 'Latency (s)']
out_tabel = OUTPUT_DIR / 'tabel_kualitas_respon_bab4.csv'
df_export.to_csv(out_tabel, index=False, float_format='%.4f')
print(f"\n✓ Tersimpan: {out_tabel}")


# =============================================================================
# GAMBAR 3 — KURVA PARETO INTERAKTIF (PLOTLY) BERGAYA MINIMALIS / CLEAN
# =============================================================================

x_col = 'Latency_s'
y_col = 'NDCG@10'

label_map = {
    'Base_14B':       'Base 14B',
    'Pruned_10P':     'Pruned 10%',
    'Pruned_20P':     'Pruned 20%',
    'Pruned_30P':     'Pruned 30%',
    'Pruned_10P_FT':  'Pruned + FT 10%',
    'Pruned_20P_FT':  'Pruned + FT 20%',
    'Pruned_30P_FT':  'Pruned + FT 30%',
}

# Hitung Pareto Frontier
lat_vals  = df_q[x_col].values
ndcg_vals = df_q[y_col].values
pts = sorted(zip(lat_vals, ndcg_vals, range(len(lat_vals))), key=lambda p: (p[0], -p[1]))

front_lat = []
best_y = -np.inf
for px, py, pi in pts:
    if py > best_y:
        front_lat.append((px, py, pi))
        best_y = py

front_idx_lat = [p[2] for p in front_lat]

plot_df = df_q.copy()
plot_df['IsPareto'] = plot_df.index.isin(front_idx_lat)
plot_df['DisplayLabel'] = plot_df['Model'].astype(str).map(label_map)

front_df_plotly = plot_df.iloc[front_idx_lat].sort_values(x_col, ascending=False).reset_index(drop=True)

# Build figure
fig_plotly = go.Figure()
color_pareto     = '#1B63F2'
color_non_pareto = '#D3D3D3'

# Layer 1: Garis Pareto frontier (spline halus)
fig_plotly.add_trace(go.Scatter(
    x=front_df_plotly[x_col],
    y=front_df_plotly[y_col],
    mode='lines',
    line=dict(color=color_pareto, width=2.5, dash='dash',
              shape='spline', smoothing=0.8),
    hoverinfo='skip', showlegend=False
))

# Layer 2: Titik tiap model
for _, row in plot_df.iterrows():
    is_pareto = row['IsPareto']
    marker_color = color_pareto if is_pareto else color_non_pareto
    fig_plotly.add_trace(go.Scatter(
        x=[row[x_col]], y=[row[y_col]],
        mode='markers+text',
        marker=dict(size=14, color=marker_color, line=dict(width=0)),
        text=[f"  {row['DisplayLabel']}"],
        textposition='middle right',
        textfont=dict(size=12, color='#444444', family='Arial'),
        name=row['DisplayLabel'],
        customdata=[[row['VRAM_GB'], row['ParseSuccess']]],
        hovertemplate=(
            f"<b>{row['DisplayLabel']}</b><br>"
            f"Latency: %{{x:.2f}} s<br>"
            f"NDCG@10: %{{y:.4f}}<br>"
            f"VRAM: %{{customdata[0]:.2f}} GB<extra></extra>"
        ),
        showlegend=False, cliponaxis=False
    ))

# Padding x-axis
x_max = plot_df[x_col].max() + 0.3
x_min = plot_df[x_col].min() - 0.9

fig_plotly.update_layout(
    title=dict(
        text='<b>Kurva Pareto Trade-off Kualitas Respons (NDCG@10) terhadap Latensi Total</b>',
        x=0.5, xanchor='center',
        font=dict(size=16, color='#222222', family='Arial')
    ),
    xaxis=dict(
        title='Total Latency (detik) - Lebih Kecil Lebih Baik →',
        range=[x_max, x_min],
        showgrid=False, zeroline=False,
        showline=True, linecolor='#AAAAAA', linewidth=1,
        ticks='outside', tickcolor='#AAAAAA'
    ),
    yaxis=dict(
        title='Kualitas (NDCG@10)',
        showgrid=False, zeroline=False,
        showline=True, linecolor='#AAAAAA', linewidth=1,
        ticks='outside', tickcolor='#AAAAAA',
        tickformat='.3f'
    ),
    plot_bgcolor='white', paper_bgcolor='white',
    width=800, height=600,
    margin=dict(l=80, r=160, t=80, b=80),
)

# Simpan HTML interaktif (selalu berhasil)
out_html = OUTPUT_DIR / 'pareto_latency_interactive.html'
fig_plotly.write_html(out_html)
print(f"✓ Tersimpan: {out_html}")

# Coba simpan PNG (perlu kaleido + Chrome/Chromium)
out_png = OUTPUT_DIR / 'pareto_latency_clean.png'
try:
    fig_plotly.write_image(out_png, scale=2)
    print(f"✓ Tersimpan: {out_png}")
except Exception as e:
    print(f"⚠ PNG export Plotly gagal: {e}")
    print(f"  Solusi: pip install -U kaleido")
    print(f"  Atau buka HTML di browser dan screenshot manual.")

# Tampilkan widget interaktif di browser default
print("\n🌐 Membuka kurva Pareto interaktif di browser...")
fig_plotly.show()

print("\n" + "=" * 70)
print("✅ SELESAI! Semua output ada di folder:", OUTPUT_DIR)
print("=" * 70)