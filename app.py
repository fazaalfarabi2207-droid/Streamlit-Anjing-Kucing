"""
Aplikasi Streamlit: Deteksi Objek (YOLOv8) dan Klasifikasi Gambar (MobileNetV2)
Kelas: Anjing dan Kucing

Laporan IV Praktikum Pemrograman Big Data
Faza Al-Farabi (2308108010032)
"""

import os

# Model .h5 dibuat dengan Keras 2 (tf-keras). Variabel ini harus diatur SEBELUM TensorFlow diimpor.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
YOLO_PATH = BASE_DIR / "model" / "best.pt"
CLS_PATH = BASE_DIR / "model" / "classifier_model.h5"
SAMPLE_DIR = BASE_DIR / "sample_images"

CLASS_NAMES = ["Anjing", "Kucing"]  # urutan indeks 0 dan 1, sama dengan data.yaml
CLS_INPUT_SIZE = (224, 224)

MODE_YOLO = "Deteksi Objek (YOLOv8)"
MODE_CLS = "Klasifikasi Gambar (MobileNetV2)"
MODE_BOTH = "Keduanya (bandingkan)"

st.set_page_config(page_title="Deteksi & Klasifikasi Anjing-Kucing", page_icon="🐾", layout="wide")


# ----------------------------------------------------------------------------
# Fungsi bantu
# ----------------------------------------------------------------------------
def tampil_gambar(gambar, caption=None):
    """Tampilkan gambar selebar kolom (kompatibel dengan berbagai versi Streamlit)."""
    try:
        st.image(gambar, caption=caption, width="stretch")
    except TypeError:
        st.image(gambar, caption=caption, use_container_width=True)


@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def muat_yolo():
    from ultralytics import YOLO

    return YOLO(str(YOLO_PATH))


@st.cache_resource(show_spinner="Memuat model klasifikasi...")
def muat_klasifikasi():
    import tensorflow as tf

    return tf.keras.models.load_model(str(CLS_PATH), compile=False)


def siapkan_gambar(sumber) -> Image.Image:
    """Buka gambar, perbaiki rotasi EXIF, dan ubah ke RGB (PNG RGBA tidak error)."""
    img = Image.open(sumber)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def jalankan_yolo(img: Image.Image, conf: float):
    model = muat_yolo()
    hasil = model.predict(img, conf=conf, verbose=False)[0]
    gambar_hasil = hasil.plot()[..., ::-1]  # plot() berformat BGR -> ubah ke RGB

    baris = []
    for kotak in hasil.boxes:
        x1, y1, x2, y2 = kotak.xyxy[0].tolist()
        baris.append(
            {
                "Kelas": hasil.names[int(kotak.cls[0])],
                "Confidence": round(float(kotak.conf[0]), 3),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }
        )
    return gambar_hasil, pd.DataFrame(baris)


def jalankan_klasifikasi(img: Image.Image):
    model = muat_klasifikasi()
    arr = np.asarray(img.resize(CLS_INPUT_SIZE), dtype="float32") / 255.0
    probabilitas = model.predict(arr[np.newaxis, ...], verbose=0)[0]
    return probabilitas


def tampilkan_hasil_yolo(img: Image.Image, conf: float):
    gambar_hasil, tabel = jalankan_yolo(img, conf)
    tampil_gambar(gambar_hasil, caption="Hasil deteksi")
    if tabel.empty:
        st.info("Tidak ada objek terdeteksi. Coba turunkan nilai confidence di sidebar.")
    else:
        jumlah = tabel["Kelas"].value_counts().to_dict()
        st.success("Terdeteksi: " + ", ".join(f"{v} {k}" for k, v in jumlah.items()))
        st.dataframe(tabel, hide_index=True)


def tampilkan_hasil_klasifikasi(img: Image.Image):
    prob = jalankan_klasifikasi(img)
    indeks = int(np.argmax(prob))
    st.metric("Prediksi kelas", CLASS_NAMES[indeks], f"{prob[indeks] * 100:.1f}% yakin", delta_color="off")
    tabel = pd.DataFrame({"Kelas": CLASS_NAMES, "Probabilitas": prob.astype(float)}).set_index("Kelas")
    st.bar_chart(tabel, y="Probabilitas")
    st.caption("Klasifikasi memberi satu label untuk seluruh gambar. Untuk gambar yang berisi lebih dari satu hewan, gunakan mode deteksi.")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🐾 Pengaturan")
mode = st.sidebar.radio("Pilih mode", [MODE_YOLO, MODE_CLS, MODE_BOTH])

conf = 0.25
if mode in (MODE_YOLO, MODE_BOTH):
    conf = st.sidebar.slider(
        "Confidence minimum (YOLO)", 0.05, 0.95, 0.25, 0.05,
        help="Objek dengan keyakinan di bawah nilai ini tidak ditampilkan.",
    )

sumber_gambar = st.sidebar.radio("Sumber gambar", ["Unggah gambar", "Gunakan gambar contoh"])

st.sidebar.divider()
st.sidebar.caption("Laporan IV Praktikum Pemrograman Big Data\n\nFaza Al-Farabi (2308108010032)")

# ----------------------------------------------------------------------------
# Halaman utama
# ----------------------------------------------------------------------------
st.title("Deteksi Objek & Klasifikasi Gambar: Anjing vs Kucing")
st.write(
    "Unggah foto anjing atau kucing, lalu pilih mode di sidebar: **deteksi objek** "
    "(kotak pembatas dengan YOLOv8n) atau **klasifikasi gambar** (MobileNetV2)."
)

with st.expander("Tentang model"):
    st.markdown(
        """
- **Deteksi objek:** YOLOv8n yang dilatih pada dataset *Cats and Dogs* (Kaggle) hasil anotasi Roboflow, 2 kelas (Anjing, Kucing).
- **Klasifikasi gambar:** MobileNetV2 (*transfer learning* dari ImageNet) dengan input 224×224 dan 2 kelas.
- Dataset: 275 gambar (train 70%, valid 20%, test 10%).
        """
    )

# Pastikan berkas model ada
perlu = {MODE_YOLO: [YOLO_PATH], MODE_CLS: [CLS_PATH], MODE_BOTH: [YOLO_PATH, CLS_PATH]}[mode]
hilang = [p for p in perlu if not p.exists()]
if hilang:
    st.error("Berkas model tidak ditemukan: " + ", ".join(str(p.relative_to(BASE_DIR)) for p in hilang))
    st.stop()

# Ambil gambar
gambar = None
if sumber_gambar == "Unggah gambar":
    berkas = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png", "webp"])
    if berkas is not None:
        gambar = siapkan_gambar(berkas)
else:
    contoh = sorted(p.name for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not contoh:
        st.warning("Folder sample_images kosong.")
    else:
        pilihan = st.selectbox("Pilih gambar contoh", contoh)
        gambar = siapkan_gambar(SAMPLE_DIR / pilihan)

if gambar is None:
    st.info("Belum ada gambar. Unggah gambar atau pilih gambar contoh di sidebar.")
    st.stop()

# Tampilkan hasil
if mode == MODE_YOLO:
    kol1, kol2 = st.columns(2)
    with kol1:
        st.subheader("Gambar asli")
        tampil_gambar(gambar)
    with kol2:
        st.subheader("Hasil deteksi")
        tampilkan_hasil_yolo(gambar, conf)

elif mode == MODE_CLS:
    kol1, kol2 = st.columns(2)
    with kol1:
        st.subheader("Gambar asli")
        tampil_gambar(gambar)
    with kol2:
        st.subheader("Hasil klasifikasi")
        tampilkan_hasil_klasifikasi(gambar)

else:
    kol1, kol2, kol3 = st.columns(3)
    with kol1:
        st.subheader("Gambar asli")
        tampil_gambar(gambar)
    with kol2:
        st.subheader("Deteksi (YOLOv8)")
        tampilkan_hasil_yolo(gambar, conf)
    with kol3:
        st.subheader("Klasifikasi (MobileNetV2)")
        tampilkan_hasil_klasifikasi(gambar)
