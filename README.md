# Olah Data Bab 4 PKM RE 2026 — Versi Local Python

Kumpulan script untuk mengolah hasil eksperimen PKM RE 2026 dan menghasilkan
visualisasi yang siap dipakai di laporan akhir.

## Struktur Folder

```
project/
├── data/
│   ├── summary_inference_metrics_llm_inference_oracle_top15_full50_v1.csv
│   └── summary_quality_metrics_llm_only_oracle_ndcg_full50_v2_top15.csv
├── outputs/                  (otomatis dibuat saat script dijalankan)
├── bab4_1_inference.py       (Bab 4.1: bar chart efisiensi inferensi)
├── bab4_kualitas_pareto.py   (Bab 4.2: bar chart kualitas + Pareto Latency)
├── bab4_3_pareto_4kurva.py   (Bab 4.3: 4 kurva Pareto trade-off)
├── requirements.txt
└── README.md
```

## Setup (Sekali Saja)

### 1. Pastikan Python sudah terinstall
Cek versi Python di terminal:
```bash
python --version
```
Minimal Python 3.9.

### 2. (Opsional tapi disarankan) Buat virtual environment

**Windows PowerShell:**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install library yang dibutuhkan
```bash
pip install -r requirements.txt
```

### 4. Letakkan kedua file CSV
Letakkan kedua file CSV di folder `data/`:
- `summary_inference_metrics_llm_inference_oracle_top15_full50_v1.csv`
- `summary_quality_metrics_llm_only_oracle_ndcg_full50_v2_top15.csv`

## Cara Menjalankan

Jalankan tiap script sesuai kebutuhan. Tidak ada urutan wajib, semua berdiri sendiri.

### Bab 4.1 — Efisiensi Komputasi dan Kecepatan Inferensi

```bash
python bab4_1_inference.py
```

**CSV yang dibutuhkan:** inference
**Output:**
- `outputs/gambar_efisiensi_inferensi.png` — bar chart 4 panel (TTFT, Total Latency, Peak VRAM, Throughput)
- `outputs/tabel_efisiensi_inferensi.csv`

### Bab 4.2 — Kualitas Respon Rekomendasi

```bash
python bab4_kualitas_pareto.py
```

**CSV yang dibutuhkan:** quality
**Output:**
- `outputs/gambar_kualitas_respon.png` — bar chart 4 panel (NDCG, Recall, Hit Rate, Precision)
- `outputs/gambar_pareto_ndcg_vs_latency.png` — Pareto matplotlib
- `outputs/pareto_latency_interactive.html` — Pareto interaktif (Plotly)
- `outputs/pareto_latency_clean.png` — Pareto Plotly versi PNG
- `outputs/tabel_kualitas_respon_bab4.csv`

### Bab 4.3 — Kurva Pareto Trade-off Multi-Dimensi

```bash
python bab4_3_pareto_4kurva.py
```

**CSV yang dibutuhkan:** inference DAN quality (keduanya)
**Output (8 file):**
- `outputs/pareto_1_ndcg_vs_latency_interactive.html` + `.png`
- `outputs/pareto_2_ndcg_vs_ttft_interactive.html` + `.png`
- `outputs/pareto_3_ndcg_vs_throughput_interactive.html` + `.png`
- `outputs/pareto_4_ndcg_vs_vram_interactive.html` + `.png`

## Ringkasan Output untuk Laporan PKM RE

| Bab | File untuk Laporan |
|---|---|
| 4.1 | `gambar_efisiensi_inferensi.png` (Gambar 1) |
| 4.2 | `gambar_kualitas_respon.png` (Gambar 2) |
| 4.3 | `pareto_1_*_clean.png` (Gambar 3: NDCG vs Latency) |
| 4.3 | `pareto_2_*_clean.png` (Gambar 4: NDCG vs TTFT) |
| 4.3 | `pareto_3_*_clean.png` (Gambar 5: NDCG vs Throughput) |
| 4.3 | `pareto_4_*_clean.png` (Gambar 6: NDCG vs VRAM) |

File `.html` interaktif berguna untuk demo saat sidang/presentasi.

## Troubleshooting

### Error: "File CSV tidak ditemukan"
Pastikan struktur folder sudah benar. Atau ubah variabel `CSV_PATH` di awal script.

### Error: "ModuleNotFoundError"
Library belum terinstall. Pastikan venv sudah aktif, lalu jalankan:
```bash
pip install -r requirements.txt
```

### PNG export Plotly gagal
`kaleido` butuh Chrome/Chromium terinstall di sistem.

Solusi paling mudah: buka file HTML interaktif di browser, klik icon kamera di pojok kanan atas widget untuk download sebagai PNG manual.

Solusi lain (Windows): jalankan `plotly_get_chrome` di terminal.

### PowerShell Execution Policy error saat aktivasi venv
Jalankan sekali saja:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```z