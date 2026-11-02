# Deteksi Objek & Klasifikasi Gambar: Anjing vs Kucing

Aplikasi web berbasis **Streamlit** untuk Laporan IV Praktikum Pemrograman Big Data.
Pengguna dapat mengunggah foto anjing atau kucing, lalu memilih:

- **Deteksi objek** dengan YOLOv8n (kotak pembatas + confidence),
- **Klasifikasi gambar** dengan MobileNetV2 (probabilitas tiap kelas),
- atau **keduanya** sekaligus untuk dibandingkan.

**Demo:** _(isi dengan link Streamlit Cloud setelah deploy)_

## Model

| Model | Berkas | Keterangan |
|---|---|---|
| Deteksi objek | `model/best.pt` | YOLOv8n, 100 epoch, 2 kelas (Anjing, Kucing) |
| Klasifikasi gambar | `model/classifier_model.h5` | MobileNetV2 (transfer learning ImageNet), input 224×224, 2 kelas |

Dataset: *Cats and Dogs* (Kaggle), 275 gambar dianotasi di Roboflow (project "Laporan 3 Faza"),
dibagi train 70% / valid 20% / test 10% dan diekspor dalam format YOLOv8.

## Struktur Proyek

```
.
├── app.py               # aplikasi Streamlit
├── requirements.txt     # daftar library Python
├── packages.txt         # paket sistem untuk Streamlit Cloud
├── model/
│   ├── best.pt
│   └── classifier_model.h5
└── sample_images/       # contoh gambar untuk dicoba
```

## Menjalankan di Komputer Sendiri

```bash
git clone https://github.com/USERNAME/NAMA-REPO.git
cd NAMA-REPO
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## Penulis

Faza Al-Farabi (2308108010032), Statistika, Universitas Syiah Kuala.
