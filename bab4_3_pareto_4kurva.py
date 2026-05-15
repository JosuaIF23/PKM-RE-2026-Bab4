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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


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


def catmull_rom_spline(xs, ys, samples_per_segment=40):
    """Buat garis halus sederhana tanpa dependensi SciPy."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if len(xs) < 3:
        return xs, ys

    x_pad = np.concatenate(([xs[0]], xs, [xs[-1]]))
    y_pad = np.concatenate(([ys[0]], ys, [ys[-1]]))
    x_smooth = []
    y_smooth = []

    for i in range(1, len(x_pad) - 2):
        p0 = np.array([x_pad[i - 1], y_pad[i - 1]])
        p1 = np.array([x_pad[i], y_pad[i]])
        p2 = np.array([x_pad[i + 1], y_pad[i + 1]])
        p3 = np.array([x_pad[i + 2], y_pad[i + 2]])

        for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2 * p1) +
                (-p0 + p2) * t +
                (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            x_smooth.append(point[0])
            y_smooth.append(point[1])

    x_smooth.append(xs[-1])
    y_smooth.append(ys[-1])
    return np.asarray(x_smooth), np.asarray(y_smooth)


def compute_label_yshifts(y_values, y_range_layout, usable_height_px, min_gap_px=22):
    """Atur jarak label vertikal dalam pixel agar tidak saling bertumpuk."""
    y_values = np.asarray(y_values, dtype=float)
    y_min, y_max = min(y_range_layout), max(y_range_layout)
    denom = max(y_max - y_min, 1e-9)

    base_positions = (y_values - y_min) / denom * usable_height_px
    order = np.argsort(base_positions)
    adjusted = base_positions.copy()

    for idx in range(1, len(order)):
        cur = order[idx]
        prev = order[idx - 1]
        adjusted[cur] = max(adjusted[cur], adjusted[prev] + min_gap_px)

    overflow = adjusted[order[-1]] - usable_height_px
    if overflow > 0:
        adjusted -= overflow

    underflow = adjusted[order[0]]
    if underflow < 0:
        adjusted -= underflow

    return adjusted - base_positions


def compute_label_positions(xs, ys, x_range_layout, y_range_layout, reverse_x=False):
    """Hitung posisi label yang tetap dekat dengan titik tetapi tidak saling bertumpuk."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    x_left = min(x_range_layout)
    x_right = max(x_range_layout)
    y_bottom = min(y_range_layout)
    y_top = max(y_range_layout)
    x_span = max(x_right - x_left, 1e-9)
    y_span = max(y_top - y_bottom, 1e-9)

    y_gap = max(y_span * 0.055, 0.0018)
    order = np.argsort(ys)
    label_ys = ys.copy()

    for idx in range(1, len(order)):
        cur = order[idx]
        prev = order[idx - 1]
        label_ys[cur] = max(label_ys[cur], label_ys[prev] + y_gap)

    overflow = label_ys[order[-1]] - (y_top - y_gap * 0.25)
    if overflow > 0:
        label_ys -= overflow

    underflow = (y_bottom + y_gap * 0.25) - label_ys[order[0]]
    if underflow > 0:
        label_ys += underflow

    x_offset = x_span * 0.018
    if reverse_x:
        label_xs = xs - x_offset
    else:
        label_xs = xs + x_offset

    return label_xs, label_ys


def export_static_png_fallback(plot_df, front_df, x_col, y_col, x_label, y_label,
                               title, x_range_layout, y_range_layout, out_png,
                               color_pareto, color_non_pareto, label_yshifts,
                               y_tickvals, x_fmt):
    """Fallback PNG statis jika export Plotly gagal di environment lokal."""
    fig, ax = plt.subplots(figsize=(9.4, 8.2), dpi=200)
    smooth_x, smooth_y = catmull_rom_spline(front_df[x_col], front_df[y_col])

    ax.plot(
        smooth_x,
        smooth_y,
        color=color_pareto,
        linewidth=2.2,
        linestyle='--',
        zorder=2
    )

    for idx, row in plot_df.iterrows():
        marker_color = color_pareto if row['IsPareto'] else color_non_pareto
        ax.vlines(
            row[x_col], y_range_layout[0], row[y_col],
            colors='#D6DEEF', linestyles='--', linewidth=0.9, zorder=1
        )
        ax.hlines(
            row[y_col], x_range_layout[0], row[x_col],
            colors='#D6DEEF', linestyles='--', linewidth=0.9, zorder=1
        )
        ax.scatter(
            row[x_col],
            row[y_col],
            s=95,
            color=marker_color,
            edgecolors='none',
            zorder=3
        )
        ax.annotate(
            row['DisplayLabel'],
            (row[x_col], row[y_col]),
            xytext=(10, float(label_yshifts[idx]) * 0.55),
            textcoords='offset points',
            ha='left',
            va='center',
            fontsize=10,
            color='#444444',
            zorder=4
        )

    ax.set_title(title, fontsize=14, color='#222222', pad=14)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_xlim(x_range_layout[0], x_range_layout[1])
    ax.set_ylim(y_range_layout[0], y_range_layout[1])
    ax.set_yticks(y_tickvals)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter(x_fmt.replace(':', '%')))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#AAAAAA')
    ax.spines['bottom'].set_color('#AAAAAA')
    ax.tick_params(axis='both', colors='#555555', labelsize=10)
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.8, color='#D9DFEA')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


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
            mode='markers',
            marker=dict(size=14, color=marker_color, line=dict(width=0)),
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
        plot_x_max = x_data_max + x_range * 0.06
        plot_x_min = x_data_min - x_range * 0.12
        x_range_layout = [plot_x_max, plot_x_min]
    else:
        plot_x_max = x_data_max + x_range * 0.12
        plot_x_min = x_data_min - x_range * 0.06
        x_range_layout = [plot_x_min, plot_x_max]

    # Padding sumbu Y
    y_data_min = plot_df[y_col].min()
    y_data_max = plot_df[y_col].max()
    y_range = y_data_max - y_data_min
    if y_range == 0:
        y_range = abs(y_data_max) * 0.1 + 0.01

    # Perketat skala Y agar perbedaan NDCG lebih terbaca.
    y_padding = max(y_range * 0.028, 0.0022)
    y_range_layout = [y_data_min - y_padding, y_data_max + y_padding]
    y_tickvals = np.sort(plot_df[y_col].unique())
    label_yshifts = compute_label_yshifts(
        plot_df[y_col].values,
        y_range_layout=y_range_layout,
        usable_height_px=500,
        min_gap_px=14
    )

    for idx, row in plot_df.iterrows():
        fig.add_shape(
            type='line',
            x0=row[x_col], y0=y_range_layout[0],
            x1=row[x_col], y1=row[y_col],
            line=dict(color='#D6DEEF', width=1, dash='dot'),
            layer='below'
        )
        fig.add_shape(
            type='line',
            x0=x_range_layout[0], y0=row[y_col],
            x1=row[x_col], y1=row[y_col],
            line=dict(color='#D6DEEF', width=1, dash='dot'),
            layer='below'
        )
        fig.add_annotation(
            x=row[x_col],
            y=row[y_col],
            text=row['DisplayLabel'],
            xref='x',
            yref='y',
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            xshift=12,
            yshift=float(label_yshifts[idx]),
            font=dict(size=12, color='#444444', family='Arial'),
            align='left'
        )

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            x=0.5, xanchor='center',
            font=dict(size=15, color='#222222', family='Arial')
        ),
        xaxis=dict(
            title=x_label,
            range=x_range_layout,
            tickformat=x_fmt[1:],
            nticks=6,
            showgrid=False,
            zeroline=False,
            showline=True, linecolor='#AAAAAA', linewidth=1,
            ticks='outside', tickcolor='#AAAAAA',
            automargin=True
        ),
        yaxis=dict(
            title=y_label,
            range=y_range_layout,
            tickvals=y_tickvals.tolist(),
            showgrid=True, gridcolor='#D9DFEA', griddash='dash', gridwidth=0.8,
            zeroline=False,
            showline=True, linecolor='#AAAAAA', linewidth=1,
            ticks='outside', tickcolor='#AAAAAA',
            tickformat='.4f',
            automargin=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=920, height=780,
        margin=dict(l=85, r=190, t=85, b=85),
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

    if not out_png.exists() or out_png.stat().st_mtime < out_html.stat().st_mtime:
        export_static_png_fallback(
            plot_df=plot_df,
            front_df=front_df_p,
            x_col=x_col,
            y_col=y_col,
            x_label=x_label,
            y_label=y_label,
            title=title,
            x_range_layout=x_range_layout,
            y_range_layout=y_range_layout,
            out_png=out_png,
            color_pareto=color_pareto,
            color_non_pareto=color_non_pareto,
            label_yshifts=label_yshifts,
            y_tickvals=y_tickvals,
            x_fmt=x_fmt,
        )
        print("Fallback PNG matplotlib diperbarui agar sinkron dengan HTML terbaru")

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
