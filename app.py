import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- KONFIGURASI DAN MODEL ---
st.set_page_config(
    page_title="Deteksi Penyakit Daun Tomat",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="auto",
)

# Sesuaikan dengan nama file model Anda
MODEL_PATH = "tomato_model.keras"

# Daftar nama kelas dari kode versi kedua Anda
CLASS_NAMES = [
    "Tomato_healthy",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Bacterial_spot",
    "Tomato_Septoria_leaf_spot",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
]
# ---------------------------

# --- KUSTOMISASI CSS ---
st.markdown(
    """
<style>
/* Membuat kartu untuk hasil */
.card {
    background-color: #262730; /* Warna latar belakang kartu lebih gelap */
    border-radius: 15px;
    padding: 25px;
    margin-top: 20px;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
}
.card:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}
/* Mengubah tampilan tombol */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #4CAF50;
    color: #4CAF50;
    background-color: transparent;
    transition: 0.3s;
}
.stButton>button:hover {
    border: 1px solid #4CAF50;
    color: white;
    background-color: #4CAF50;
}
.stButton>button:focus {
    color: white !important;
    background-color: #4CAF50 !important;
    border: 1px solid #4CAF50 !important;
}
/* Menyembunyikan elemen default Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)
# ---------------------------


# --- FUNGSI-FUNGSI ---
@st.cache_resource
def load_model():
    """Memuat model Keras dan menangani error."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None


model = load_model()


def predict(image_data):
    """Memproses gambar dan mengembalikan prediksi & kepercayaan."""
    if model is None:
        return "Model tidak berhasil dimuat.", 0.0

    size = (224, 224)
    image = Image.open(image_data).convert("RGB").resize(size)
    img_array = np.expand_dims(np.array(image), axis=0)

    predictions = model.predict(img_array)
    predicted_class_name = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    # Membersihkan nama kelas untuk tampilan yang lebih baik
    cleaned_class_name = predicted_class_name.replace("_", " ").replace("  ", " ")
    return cleaned_class_name, confidence


# ---------------------------


# --- TATA LETAK APLIKASI ---
st.title("🍅 Detektor Penyakit Daun Tomat")
st.markdown(
    "Unggah gambar daun tomat untuk dianalisis oleh AI. Aplikasi ini akan mengidentifikasi jika daun tersebut sehat atau terjangkit salah satu dari 9 penyakit umum."
)
st.divider()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("Langkah 1: Unggah Gambar")
    uploaded_file = st.file_uploader(
        "Pilih file gambar (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # PERBAIKAN: Menggunakan `uploaded_file.name` sebagai ganti `uploaded_file.id`
        if st.session_state.get("last_uploaded_name") != uploaded_file.name:
            st.session_state["last_uploaded_name"] = uploaded_file.name
            if "result" in st.session_state:
                del st.session_state["result"]

        st.image(uploaded_file, caption="Gambar Anda", use_container_width=True)

        if st.button("✨ Analisis Gambar Ini"):
            with st.spinner("AI sedang bekerja... Menganalisis piksel demi piksel..."):
                predicted_class, confidence = predict(uploaded_file)
                st.session_state["result"] = (predicted_class, confidence)

with col2:
    st.header("Langkah 2: Hasil Analisis")

    if "result" in st.session_state:
        predicted_class, confidence = st.session_state["result"]

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)

            if "healthy" in predicted_class.lower():
                st.success(f"✅ Daun Terdeteksi **{predicted_class}**")
            else:
                st.warning(f"⚠️ Daun Terdeteksi **{predicted_class}**")

            st.metric(label="Tingkat Kepercayaan AI", value=f"{confidence:.2f}%")

            st.progress(int(confidence))

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(
            "Hasil analisis akan muncul di sini setelah Anda mengunggah gambar dan menekan tombol analisis."
        )

# Expander untuk informasi tambahan
with st.expander("ℹ️ Tentang Aplikasi Ini"):
    st.write(
        """
        Aplikasi ini menggunakan model *Convolutional Neural Network* (CNN) dengan optimasi Hyperparameter yang dilatih pada dataset PlantVillage.
        Hasil prediksi bersifat indikatif, konsultasikan dengan ahli pertanian untuk penanganan lebih lanjut.
        """
    )
