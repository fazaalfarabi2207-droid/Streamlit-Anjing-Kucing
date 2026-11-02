"""
Anjing atau Kucing? Aplikasi Streamlit untuk deteksi objek (YOLOv8)
dan klasifikasi gambar (MobileNetV2).

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
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
YOLO_PATH = BASE_DIR / "model" / "best.pt"
CLS_PATH = BASE_DIR / "model" / "classifier_model.h5"
SAMPLE_DIR = BASE_DIR / "sample_images"

CLASS_NAMES = ["Anjing", "Kucing"]  # indeks 0 dan 1, sama dengan data.yaml
CLS_INPUT_SIZE = (224, 224)
MAX_SISI = 1600  # gambar yang lebih besar dikecilkan agar proses tetap cepat

# Satu warna untuk satu kelas, dipakai konsisten di kotak deteksi, label, dan grafik.
WARNA = {"Anjing": "#FFB224", "Kucing": "#B794F6"}
WARNA_LAIN = "#9AA9BC"
INK = "#0E1A2B"

MODE_DETEKSI = "Deteksi"
MODE_KLASIFIKASI = "Klasifikasi"
MODE_BANDINGKAN = "Keduanya"
PENJELASAN_MODE = {
    MODE_DETEKSI: "YOLOv8n menandai lokasi setiap hewan dengan kotak dan tingkat keyakinan.",
    MODE_KLASIFIKASI: "MobileNetV2 memberi satu label untuk seluruh foto, lengkap dengan probabilitasnya.",
    MODE_BANDINGKAN: "Jalankan kedua model pada foto yang sama, lalu lihat apakah hasilnya sepakat.",
}

SUMBER_UNGGAH = "Unggah foto"
SUMBER_CONTOH = "Pilih contoh"

st.set_page_config(page_title="Anjing atau Kucing?", page_icon="🐾", layout="wide")

# ----------------------------------------------------------------------------
# Tampilan (CSS)
# ----------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap');

:root {
  --dog: #FFB224; --cat: #B794F6; --ink: #0E1A2B;
  --panel: #16263B; --line: #2A3D56; --text: #E9EEF5; --muted: #9AA9BC;
}

.stApp, .stApp p, .stApp label, .stApp button, .stApp input,
.stApp li, .stApp a, .stApp small {
  font-family: 'Instrument Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp [data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1120px; padding-top: 2.6rem; padding-bottom: 4rem;
}
footer { visibility: hidden; }

.mark { display: block; margin-bottom: 1.1rem; }
.hero-title {
  font-family: 'Bricolage Grotesque', 'Instrument Sans', system-ui, sans-serif;
  font-weight: 800; font-size: clamp(2.7rem, 7vw, 4.8rem); line-height: 1;
  letter-spacing: -0.035em; margin: 0 0 1rem; color: var(--text);
}
.hero-text { max-width: 34rem; color: var(--muted); font-size: 1.1rem; line-height: 1.55; margin: 0 0 1.3rem; }
.legend { display: flex; gap: .6rem; flex-wrap: wrap; }
.chip {
  display: inline-flex; align-items: center; gap: .5rem; padding: .3rem .85rem .3rem .65rem;
  border: 1px solid var(--line); border-radius: 999px; font-weight: 600; font-size: .92rem; color: var(--text);
}
.chip i { width: .7rem; height: .7rem; border-radius: 50%; display: inline-block; background: var(--c); }

.sect {
  font-family: 'Bricolage Grotesque', 'Instrument Sans', system-ui, sans-serif;
  font-weight: 600; font-size: 1.45rem; letter-spacing: -0.01em; color: var(--text);
  margin: 2.6rem 0 .9rem; padding-top: 1.4rem; border-top: 1px solid var(--line);
}
.subtle { color: var(--muted); font-size: .98rem; margin: .3rem 0 1rem; }

[data-testid="stImage"] img { border-radius: 14px; }

.verdict { border-left: 6px solid var(--c); padding: .15rem 0 .15rem 1.1rem; margin: 0 0 1.1rem; }
.verdict-label { color: var(--muted); font-size: .98rem; margin: 0; }
.verdict-name {
  font-family: 'Bricolage Grotesque', 'Instrument Sans', system-ui, sans-serif;
  font-weight: 800; font-size: clamp(2.5rem, 5.5vw, 3.8rem); line-height: 1.05;
  letter-spacing: -0.035em; color: var(--c); margin: .1rem 0 0;
}
.bars { display: grid; gap: .75rem; margin: 0 0 1.2rem; }
.bar { display: grid; grid-template-columns: 5rem 1fr 4.4rem; align-items: center; gap: .8rem; }
.bar-name { font-weight: 600; color: var(--text); }
.track { height: .85rem; border-radius: 999px; background: var(--panel); overflow: hidden; }
.fill { height: 100%; border-radius: 999px; background: var(--c); }
.bar-val { text-align: right; font-variant-numeric: tabular-nums; color: var(--text); }

.block-title { color: var(--muted); font-size: .98rem; margin: 0 0 .5rem; }
.found { display: flex; flex-wrap: wrap; gap: .5rem; margin: 0 0 1.2rem; }
.tag {
  background: var(--c); color: var(--ink); font-weight: 700; border-radius: 8px;
  padding: .3rem .7rem; font-size: 1rem;
}
.note {
  background: var(--panel); border-left: 4px solid var(--muted); border-radius: 10px;
  padding: .9rem 1.1rem; color: var(--text); line-height: 1.5; margin-top: .4rem;
}
.note.sepakat { border-left-color: var(--c); }
.note { margin-bottom: 1rem; }

.footer { color: var(--muted); font-size: .92rem; line-height: 1.6; margin-top: 3.5rem; padding-top: 1.2rem; border-top: 1px solid var(--line); }
"""


def html(s: str) -> str:
    """Rapatkan HTML satu baris agar tidak dianggap blok kode oleh Markdown."""
    return "".join(baris.strip() for baris in s.strip().splitlines())


st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Fungsi bantu: tampilan
# ----------------------------------------------------------------------------
def tampil_gambar(gambar, caption=None):
    """Tampilkan gambar selebar kolom (kompatibel dengan berbagai versi Streamlit)."""
    try:
        st.image(gambar, caption=caption, width="stretch")
    except TypeError:
        st.image(gambar, caption=caption, use_container_width=True)


def tombol(label, **kwargs):
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, use_container_width=True, **kwargs)


def persen(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",") + "%"


def judul_bagian(teks: str):
    st.markdown(html(f'<div class="sect" role="heading" aria-level="2">{teks}</div>'), unsafe_allow_html=True)


def html_verdict(kelas: str, prob) -> str:
    batang = "".join(
        f'<div class="bar" style="--c:{WARNA[n]}"><span class="bar-name">{n}</span>'
        f'<div class="track"><div class="fill" style="width:{p * 100:.1f}%"></div></div>'
        f'<span class="bar-val">{persen(float(p))}</span></div>'
        for n, p in zip(CLASS_NAMES, prob)
    )
    return html(
        f'<div class="verdict" style="--c:{WARNA[kelas]}"><p class="verdict-label">MobileNetV2 menebak</p>'
        f'<div class="verdict-name">{kelas}</div></div><div class="bars">{batang}</div>'
    )


def html_temuan(deteksi: list) -> str:
    tag = "".join(
        f'<span class="tag" style="--c:{WARNA.get(d["Kelas"], WARNA_LAIN)}">{d["Kelas"]} {d["Confidence"] * 100:.0f}%</span>'
        for d in deteksi
    )
    return html(f'<p class="block-title">YOLOv8 menemukan {len(deteksi)} objek</p><div class="found">{tag}</div>')


# ----------------------------------------------------------------------------
# Fungsi bantu: model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def muat_yolo():
    from ultralytics import YOLO

    return YOLO(str(YOLO_PATH))


@st.cache_resource(show_spinner="Memuat model klasifikasi...")
def muat_klasifikasi():
    import tensorflow as tf

    return tf.keras.models.load_model(str(CLS_PATH), compile=False)


def siapkan_gambar(sumber) -> Image.Image:
    """Buka gambar, perbaiki rotasi EXIF, ubah ke RGB, dan kecilkan jika terlalu besar."""
    img = Image.open(sumber)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SISI, MAX_SISI))
    return img


@st.cache_data(show_spinner=False)
def thumbnail(path: str, sisi: int = 300) -> Image.Image:
    """Potongan persegi supaya galeri contoh rapi."""
    return ImageOps.fit(siapkan_gambar(path), (sisi, sisi))


def jalankan_yolo(img: Image.Image, conf: float) -> list:
    hasil = muat_yolo().predict(img, conf=conf, verbose=False)[0]
    lebar, tinggi = img.size
    deteksi = []
    for kotak in hasil.boxes:
        x1, y1, x2, y2 = kotak.xyxy[0].tolist()
        deteksi.append(
            {
                "Kelas": hasil.names[int(kotak.cls[0])],
                "Confidence": round(float(kotak.conf[0]), 3),
                "x1": int(max(0, x1)),
                "y1": int(max(0, y1)),
                "x2": int(min(lebar, x2)),
                "y2": int(min(tinggi, y2)),
            }
        )
    return deteksi


def gambar_dengan_kotak(img: Image.Image, deteksi: list) -> Image.Image:
    """Gambar kotak deteksi dengan warna tetap per kelas (bukan palet bawaan YOLO)."""
    hasil = img.copy()
    pena = ImageDraw.Draw(hasil)
    sisi = min(hasil.size)
    tebal = max(3, round(sisi / 140))
    try:
        font = ImageFont.load_default(size=max(16, round(sisi / 26)))
    except TypeError:  # Pillow lama
        font = ImageFont.load_default()

    for d in deteksi:
        warna = WARNA.get(d["Kelas"], WARNA_LAIN)
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        pena.rounded_rectangle([x1, y1, x2, y2], radius=tebal * 2, outline=warna, width=tebal)

        teks = f'{d["Kelas"]} {d["Confidence"] * 100:.0f}%'
        kiri, atas, kanan, bawah = pena.textbbox((0, 0), teks, font=font)
        pad = tebal * 2
        tinggi_label = (bawah - atas) + pad * 2
        lebar_label = (kanan - kiri) + pad * 2
        y_label = y1 - tinggi_label if y1 - tinggi_label >= 0 else y1
        pena.rounded_rectangle([x1, y_label, x1 + lebar_label, y_label + tinggi_label], radius=tebal * 2, fill=warna)
        pena.text((x1 + pad - kiri, y_label + pad - atas), teks, font=font, fill=INK)
    return hasil


def jalankan_klasifikasi(img: Image.Image):
    arr = np.asarray(img.resize(CLS_INPUT_SIZE), dtype="float32") / 255.0
    return muat_klasifikasi().predict(arr[np.newaxis, ...], verbose=0)[0]


def kelas_dominan(deteksi: list):
    """Kelas dengan total luas kotak terbesar."""
    luas = {}
    for d in deteksi:
        luas[d["Kelas"]] = luas.get(d["Kelas"], 0) + (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])
    return max(luas, key=luas.get) if luas else None


def html_kesepakatan(deteksi: list, kelas_cls: str) -> str:
    dominan = kelas_dominan(deteksi)
    if dominan is None:
        return html(
            '<div class="note">YOLOv8 tidak menemukan objek pada batas keyakinan ini, '
            "jadi hanya hasil klasifikasi yang tersedia. Coba turunkan batas keyakinan.</div>"
        )
    if dominan == kelas_cls:
        return html(f'<div class="note sepakat" style="--c:{WARNA[kelas_cls]}">Kedua model sepakat: foto ini {kelas_cls}.</div>')
    return html(
        f'<div class="note">Hasil berbeda. YOLOv8 melihat {dominan} paling dominan, sedangkan klasifikasi memilih {kelas_cls}. '
        "Periksa fotonya: mungkin ada lebih dari satu hewan, atau objeknya kecil dan tertutup.</div>"
    )


def set_contoh(nama: str):
    st.session_state["contoh"] = nama


# ----------------------------------------------------------------------------
# Kepala halaman
# ----------------------------------------------------------------------------
st.markdown(
    html(
        f"""
<svg class="mark" width="58" height="34" viewBox="0 0 58 34" aria-hidden="true">
<circle cx="17" cy="17" r="15" fill="{WARNA['Anjing']}"/>
<circle cx="41" cy="17" r="15" fill="{WARNA['Kucing']}" style="mix-blend-mode:screen"/>
</svg>
<h1 class="hero-title">Anjing atau kucing?</h1>
<p class="hero-text">Unggah satu foto. YOLOv8 menandai di mana hewannya berada, MobileNetV2 menebak jenisnya.</p>
<div class="legend">
<span class="chip" style="--c:{WARNA['Anjing']}"><i></i>Anjing</span>
<span class="chip" style="--c:{WARNA['Kucing']}"><i></i>Kucing</span>
</div>
"""
    ),
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Kontrol (kiri) dan hasil (kanan)
# ----------------------------------------------------------------------------
kol_kontrol, kol_hasil = st.columns([1.2, 1.2], gap="large")

with kol_kontrol:
    judul_bagian("Pilih gambar")

    sumber = st.segmented_control(
        "Sumber gambar", [SUMBER_UNGGAH, SUMBER_CONTOH], default=SUMBER_UNGGAH, label_visibility="collapsed"
    ) or SUMBER_UNGGAH

    gambar = None
    if sumber == SUMBER_UNGGAH:
        berkas = st.file_uploader(
            "Foto anjing atau kucing", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
        )
        if berkas is not None:
            try:
                gambar = siapkan_gambar(berkas)
            except Exception:
                st.error("Berkas tidak bisa dibaca sebagai gambar. Coba unggah file JPG, PNG, atau WEBP yang lain.")
    else:
        daftar = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not daftar:
            st.warning("Folder sample_images kosong. Tambahkan gambar JPG atau PNG ke folder itu.")
        else:
            if st.session_state.get("contoh") not in {p.name for p in daftar}:
                st.session_state["contoh"] = daftar[0].name
            for baris in range(0, len(daftar), 4):
                kolom = st.columns(4, gap="small")
                for kol, p in zip(kolom, daftar[baris:baris + 4]):
                    with kol:
                        tampil_gambar(thumbnail(str(p)))
                        tombol(
                            p.stem.replace("_", " ").title(),
                            key=f"contoh_{p.name}",
                            on_click=set_contoh,
                            args=(p.name,),
                            type="primary" if p.name == st.session_state["contoh"] else "secondary",
                        )
            gambar = siapkan_gambar(SAMPLE_DIR / st.session_state["contoh"])

    judul_bagian("Pilih analisis")

    mode = st.segmented_control(
        "Jenis analisis", list(PENJELASAN_MODE), default=MODE_DETEKSI, label_visibility="collapsed"
    ) or MODE_DETEKSI
    st.markdown(f'<p class="subtle">{PENJELASAN_MODE[mode]}</p>', unsafe_allow_html=True)

    conf = 0.25
    if mode in (MODE_DETEKSI, MODE_BANDINGKAN):
        conf = st.slider(
            "Batas keyakinan deteksi", 0.05, 0.95, 0.25, 0.05,
            help="Objek dengan keyakinan di bawah angka ini tidak ditampilkan.",
        )

    # Pastikan berkas model ada
    perlu = {MODE_DETEKSI: [YOLO_PATH], MODE_KLASIFIKASI: [CLS_PATH], MODE_BANDINGKAN: [YOLO_PATH, CLS_PATH]}[mode]
    hilang = [p for p in perlu if not p.exists()]
    if hilang:
        st.error(
            "Berkas model tidak ditemukan: " + ", ".join(str(p.relative_to(BASE_DIR)) for p in hilang)
            + ". Pastikan folder model ikut terunggah ke repository."
        )
        st.stop()

with kol_hasil:
    judul_bagian("Hasil")

    if gambar is None:
        st.info("Belum ada foto. Unggah foto atau pilih salah satu contoh di sebelah kiri.")
    else:
        deteksi, prob = None, None
        if mode in (MODE_DETEKSI, MODE_BANDINGKAN):
            deteksi = jalankan_yolo(gambar, conf)
        if mode in (MODE_KLASIFIKASI, MODE_BANDINGKAN):
            prob = jalankan_klasifikasi(gambar)

        tampil_gambar(gambar_dengan_kotak(gambar, deteksi) if deteksi is not None else gambar)
        st.write("")

        if deteksi is not None:
            if deteksi:
                st.markdown(html_temuan(deteksi), unsafe_allow_html=True)
            elif mode == MODE_DETEKSI:
                st.info("Tidak ada objek terdeteksi. Coba turunkan batas keyakinan.")

        if prob is not None:
            kelas_cls = CLASS_NAMES[int(np.argmax(prob))]
            st.markdown(html_verdict(kelas_cls, prob), unsafe_allow_html=True)
            if mode == MODE_BANDINGKAN:
                st.markdown(html_kesepakatan(deteksi, kelas_cls), unsafe_allow_html=True)
            else:
                st.caption(
                    "Klasifikasi memberi satu label untuk seluruh foto. "
                    "Untuk foto dengan lebih dari satu hewan, gunakan deteksi objek."
                )

        if deteksi:
            with st.expander("Detail kotak deteksi"):
                st.dataframe(pd.DataFrame(deteksi), hide_index=True)

# ----------------------------------------------------------------------------
# Tentang
# ----------------------------------------------------------------------------
st.write("")
with st.expander("Tentang model dan data"):
    st.markdown(
        """
- **Deteksi objek:** YOLOv8n yang dilatih pada dataset *Cats and Dogs* (Kaggle), dianotasi di Roboflow, 2 kelas (Anjing, Kucing).
- **Klasifikasi gambar:** MobileNetV2 (*transfer learning* dari ImageNet), input 224×224, 2 kelas.
- **Data:** 275 gambar, dibagi train 70%, valid 20%, dan test 10%.
        """
    )

st.markdown(
    html(
        """
<div class="footer">Laporan IV Praktikum Pemrograman Big Data<br>
Faza Al-Farabi (2308108010032), Statistika, Universitas Syiah Kuala</div>
"""
    ),
    unsafe_allow_html=True,
)
