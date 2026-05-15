# =============================================================================
# OLAH DATA BAB 4.3 — PKM RE 2026 (VERSI LOCAL PYTHON)
# KURVA PARETO TRADE-OFF KUALITAS vs EFISIENSI INFERENSI (4 KURVA)
# Style: Plotly interaktif minimalis biru dengan spline halus
#
# 4 Kurva yang dihasilkan:
#   1. NDCG@10 vs Total Latency       (kualitas vs kecepatan end-to-end)
#   2. NDCG@10 vs TTFT                (kualitas vs responsivitas awal)
#   3. NDCG@10 vs Throughput          (kualitas vs efisiensi token)
#   4. NDCG@10 vs Peak VRAM           (kualitas vs efisiensi memori)
#
# CARA PAKAI:
#   1. Pastikan struktur folder:
#        project/
#          ├── data/
#          │   ├── summary_inference_metrics_llm_inference_oracle_top15_full50_v1.csv
#          │   └── summary_quality_metrics_llm_only_oracle_ndcg_full50_v2_top15.csv
#          └── bab4_3_pareto_4kurva.py  (file ini)
#   2. Install dependensi (sekali saja):
#        pip install -r requirements.txt
#   3. Jalankan:
#        python bab4_3_pareto_4kurva.py
# =============================================================================

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =============================================================================
# KONFIGURASI — sesuaikan path jika struktur folder Anda berbeda
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
CSV_INFERENCE = SCRIPT_DIR / "data" / "summary_inference_metrics_llm_inference_oracle_top15_full50_v1.csv"
CSV_QUALITY   = SCRIPT_DIR / "data" / "summary_quality_metrics_llm_only_oracle_ndcg_full50_v2_top15.csv"
OUTPUT_DIR    = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# VALIDASI INPUT
# =============================================================================

missing = []
if not CSV_INFERENCE.exists():
    missing.append(f"  - {CSV_INFERENCE.name}")
if not CSV_QUALITY.exists():
    missing.append(f"  - {CSV_QUALITY.name}")

if missing:
    print(f"❌ ERROR: File CSV berikut tidak ditemukan di folder 'data/':")
    for m in missing:
        print(m)
    print(f"\n   Folder yang dicek: {SCRIPT_DIR / 'data'}")
    print(f"   Pastikan kedua file CSV ada di folder 'data/'.")
    sys.exit(1)

print(f"📂 CSV Inference: {CSV_INFERENCE.name}")
print(f"📂 CSV Quality:   {CSV_QUALITY.name}")
print(f"💾 Output folder: {OUTPUT_DIR}\n")


# =============================================================================
# LOAD DATA
# =============================================================================

# Urutan model konsisten
order = ['Base_14B', 'Pruned_10P', 'Pruned_20P', 'Pruned_30P',
         'Pruned_10P_FT', 'Pruned_20P_FT', 'Pruned_30P_FT']

# Load Data Inference
df_inf = pd.read_csv(CSV_INFERENCE)
df_inf['Model'] = df_inf['model_name'].str.replace('_Aligned', '', regex=False)
df_inf['Model'] = pd.Categorical(df_inf['Model'], categories=order, ordered=True)
df_inf = df_inf.sort_values('Model').reset_index(drop=True)

df_inf = df_inf.rename(columns={
    'avg_ttft':                       'TTFT_s',
    'avg_end_to_end_latency':         'Latency_s',
    'avg_peak_vram_total_gb':         'VRAM_GB',
    'avg_throughput_tokens_per_sec':  'Throughput_tps',
})[['Model', 'TTFT_s', 'Latency_s', 'VRAM_GB', 'Throughput_tps']]

# Load Data Quality
df_q = pd.read_csv(CSV_QUALITY)
df_q['Model'] = df_q['model_name'].str.replace('_Aligned', '', regex=False)
df_q['Model'] = pd.Categorical(df_q['Model'], categories=order, ordered=True)
df_q = df_q.sort_values('Model').reset_index(drop=True)

df_q = df_q.rename(columns={
    'avg_repaired_llm_ndcg@10_attainable':    'NDCG@10',
    'avg_repaired_llm_recall@10_attainable':  'Recall@10',
    'avg_repaired_llm_hit_rate@10':           'HitRate@10',
    'avg_repaired_llm_precision@10':          'Precision@10',
    'parse_success_rate_strict_top10':        'ParseSuccess',
})[['Model', 'NDCG@10', 'Recall@10', 'HitRate@10', 'Precision@10', 'ParseSuccess']]

# Gabungkan ke satu DataFrame
df = df_q.merge(df_inf, on='Model', how='left')
df['Model'] = df['Model'].astype(str)

print("=" * 75)
print("Data gabungan untuk kurva Pareto trade-off")
print("=" * 75)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# =============================================================================
# FUNGSI HELPER: BUAT KURVA PARETO PLOTLY MINIMALIS BIRU
# =============================================================================

label_map = {
    'Base_14B':       'Base 14B',
    'Pruned_10P':     'Pruned 10%',
    'Pruned_20P':     'Pruned 20%',
    'Pruned_30P':     'Pruned 30%',
    'Pruned_10P_FT':  'Pruned + FT 10%',
    'Pruned_20P_FT':  'Pruned + FT 20%',
    'Pruned_30P_FT':  'Pruned + FT 30%',
}


def pareto_indices(xs, ys, maximize_x=False):
    """
    Hitung index titik-titik Pareto-optimal.
    - maximize_x=False: minimisasi X (Latency, TTFT, VRAM) + maksimisasi Y
    - maximize_x=True:  maksimisasi X (Throughput) + maksimisasi Y
    """
    if maximize_x:
        pts = sorted(zip(xs, ys, range(len(xs))), key=lambda p: (-p[0], -p[1]))
    else:
        pts = sorted(zip(xs, ys, range(len(xs))), key=lambda p: (p[0], -p[1]))
    front = []
    best_y = -np.inf
    for px, py, pi in pts:
        if py > best_y:
            front.append(pi)
            best_y = py
    return front


def build_pareto_chart(df, x_col, y_col, x_label, y_label, title,
                       reverse_x=False, x_fmt=':.4f',
                       maximize_x=False, fname_prefix='pareto',
                       show_in_browser=False):
    """
    Buat kurva Pareto interaktif Plotly bergaya minimalis biru.
    """
    # Hitung Pareto frontier
    front_idx = pareto_indices(df[x_col].values, df[y_col].values, maximize_x=maximize_x)
    plot_df = df.copy()
    plot_df['IsPareto'] = plot_df.index.isin(front_idx)
    plot_df['DisplayLabel'] = plot_df['Model'].map(label_map)

    if maximize_x:
        front_df_p = plot_df.iloc[front_idx].sort_values(x_col, ascending=True).reset_index(drop=True)
    else:
        front_df_p = plot_df.iloc[front_idx].sort_values(x_col, ascending=False).reset_index(drop=True)

    fig = go.Figure()
    color_pareto     = '#1B63F2'
    color_non_pareto = '#D3D3D3'

    # LAYER 1: Garis Pareto Frontier (spline halus)
    fig.add_trace(go.Scatter(
        x=front_df_p[x_col],
        y=front_df_p[y_col],
        mode='lines',
        line=dict(
            color=color_pareto, width=2.5, dash='dash',
            shape='spline', smoothing=0.8
        ),
        hoverinfo='skip',
        showlegend=False
    ))

    # LAYER 2: Titik per model
    for _, row in plot_df.iterrows():
        marker_color = color_pareto if row['IsPareto'] else color_non_pareto
        fig.add_trace(go.Scatter(
            x=[row[x_col]], y=[row[y_col]],
            mode='markers+text',
            marker=dict(size=14, color=marker_color, line=dict(width=0)),
            text=[f"  {row['DisplayLabel']}"],
            textposition='middle right',
            textfont=dict(size=12, color='#444444', family='Arial'),
            name=row['DisplayLabel'],
            customdata=[[row['VRAM_GB'], row['Latency_s'],
                         row['TTFT_s'], row['Throughput_tps']]],
            hovertemplate=(
                f"<b>{row['DisplayLabel']}</b><br>"
                f"{x_label}: %{{x{x_fmt}}}<br>"
                f"NDCG@10: %{{y:.4f}}<br>"
                f"─────────────<br>"
                f"Latency: %{{customdata[1]:.4f}} s<br>"
                f"TTFT: %{{customdata[2]:.4f}} s<br>"
                f"VRAM: %{{customdata[0]:.2f}} GB<br>"
                f"Throughput: %{{customdata[3]:.2f}} tok/s<extra></extra>"
            ),
            showlegend=False,
            cliponaxis=False
        ))

    # Padding sumbu X otomatis
    x_data_min = plot_df[x_col].min()
    x_data_max = plot_df[x_col].max()
    x_range = x_data_max - x_data_min
    if x_range == 0:
        x_range = abs(x_data_max) * 0.1 + 1.0

    if reverse_x:
        plot_x_max = x_data_max + x_range * 0.10
        plot_x_min = x_data_min - x_range * 0.55
        x_range_layout = [plot_x_max, plot_x_min]
    else:
        plot_x_max = x_data_max + x_range * 0.55
        plot_x_min = x_data_min - x_range * 0.10
        x_range_layout = [plot_x_min, plot_x_max]

    # Padding sumbu Y
    y_data_min = plot_df[y_col].min()
    y_data_max = plot_df[y_col].max()
    y_range = y_data_max - y_data_min
    if y_range == 0:
        y_range = abs(y_data_max) * 0.1 + 0.01

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            x=0.5, xanchor='center',
            font=dict(size=15, color='#222222', family='Arial')
        ),
        xaxis=dict(
            title=x_label,
            range=x_range_layout,
            showgrid=False, zeroline=False,
            showline=True, linecolor='#AAAAAA', linewidth=1,
            ticks='outside', tickcolor='#AAAAAA'
        ),
        yaxis=dict(
            title=y_label,
            range=[y_data_min - y_range * 0.20, y_data_max + y_range * 0.20],
            showgrid=False, zeroline=False,
            showline=True, linecolor='#AAAAAA', linewidth=1,
            ticks='outside', tickcolor='#AAAAAA',
            tickformat='.3f'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900, height=600,
        margin=dict(l=80, r=180, t=80, b=80),
    )

    # Simpan HTML interaktif
    out_html = OUTPUT_DIR / f'{fname_prefix}_interactive.html'
    fig.write_html(out_html)
    print(f"✓ HTML: {out_html.name}")

    # Coba simpan PNG
    out_png = OUTPUT_DIR / f'{fname_prefix}_clean.png'
    try:
        fig.write_image(out_png, scale=2)
        print(f"✓ PNG:  {out_png.name}")
    except Exception:
        print(f"⚠ PNG export gagal (perlu Chrome/Chromium di sistem)")
        print(f"  Solusi: buka HTML di browser, klik icon kamera di pojok kanan atas")

    # Print ringkasan Pareto
    print(f"\n  Pareto-optimal:")
    for idx in front_idx:
        r = plot_df.iloc[idx]
        print(f"    ⭐ {r['DisplayLabel']:18s} | {x_col}={r[x_col]:.4f} | NDCG={r[y_col]:.4f}")

    # Buka di browser default (opsional)
    if show_in_browser:
        fig.show()
    return fig


# =============================================================================
# KURVA 1 — NDCG@10 vs TOTAL LATENCY
# =============================================================================
print("\n" + "=" * 75)
print("KURVA 1: NDCG@10 vs TOTAL LATENCY")
print("=" * 75)
build_pareto_chart(
    df, x_col='Latency_s', y_col='NDCG@10',
    x_label='Total Latency (detik) - Lebih Kecil Lebih Baik →',
    y_label='Kualitas (NDCG@10)',
    title='Kurva Pareto Trade-off: Kualitas (NDCG@10) vs Total Latency',
    reverse_x=True, x_fmt=':.4f', maximize_x=False,
    fname_prefix='pareto_1_ndcg_vs_latency',
    show_in_browser=False  # ← tambahkan ini supaya tidak auto-open
    
)


# =============================================================================
# KURVA 2 — NDCG@10 vs TTFT
# =============================================================================
print("\n" + "=" * 75)
print("KURVA 2: NDCG@10 vs TTFT")
print("=" * 75)
build_pareto_chart(
    df, x_col='TTFT_s', y_col='NDCG@10',
    x_label='Time-to-First-Token (detik) - Lebih Kecil Lebih Baik →',
    y_label='Kualitas (NDCG@10)',
    title='Kurva Pareto Trade-off: Kualitas (NDCG@10) vs TTFT',
    reverse_x=True, x_fmt=':.4f', maximize_x=False,
    fname_prefix='pareto_2_ndcg_vs_ttft',
    show_in_browser=False  # ← tambahkan ini supaya tidak auto-open
)


# =============================================================================
# KURVA 3 — NDCG@10 vs THROUGHPUT
# =============================================================================
print("\n" + "=" * 75)
print("KURVA 3: NDCG@10 vs THROUGHPUT")
print("=" * 75)
build_pareto_chart(
    df, x_col='Throughput_tps', y_col='NDCG@10',
    x_label='Throughput (tok/s) - Lebih Besar Lebih Baik →',
    y_label='Kualitas (NDCG@10)',
    title='Kurva Pareto Trade-off: Kualitas (NDCG@10) vs Throughput',
    reverse_x=False, x_fmt=':.2f', maximize_x=True,
    fname_prefix='pareto_3_ndcg_vs_throughput',
    show_in_browser=False  # ← tambahkan ini supaya tidak auto-open
)


# =============================================================================
# KURVA 4 — NDCG@10 vs PEAK VRAM
# =============================================================================
print("\n" + "=" * 75)
print("KURVA 4: NDCG@10 vs PEAK VRAM")
print("=" * 75)
build_pareto_chart(
    df, x_col='VRAM_GB', y_col='NDCG@10',
    x_label='Peak VRAM (GB) - Lebih Kecil Lebih Baik →',
    y_label='Kualitas (NDCG@10)',
    title='Kurva Pareto Trade-off: Kualitas (NDCG@10) vs Peak VRAM',
    reverse_x=True, x_fmt=':.2f', maximize_x=False,
    fname_prefix='pareto_4_ndcg_vs_vram',
    show_in_browser=False  # ← tambahkan ini supaya tidak auto-open
)


print("\n" + "=" * 75)
print(f"✅ SELESAI! Semua output (8 file) ada di folder: {OUTPUT_DIR}")
print("=" * 75)
print("\nFile output yang dihasilkan:")
for f in sorted(OUTPUT_DIR.glob('pareto_*')):
    print(f"  - {f.name}")