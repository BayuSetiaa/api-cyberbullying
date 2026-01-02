from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import os # <-- Tambahkan import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- KONFIGURASI NLTK KHUSUS CLOUD ---
# Buat folder lokal untuk menyimpan data NLTK
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

# Beritahu NLTK untuk mencari data di folder ini
nltk.data.path.append(nltk_data_dir)

# Download data ke folder tersebut jika belum ada
try:
    nltk.data.find('tokenizers/punkt', paths=[nltk_data_dir])
except LookupError:
    print("Downloading punkt...")
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords', paths=[nltk_data_dir])
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords', download_dir=nltk_data_dir)
# ------------------------------------


# --- 1. Inisialisasi App ---
app = FastAPI(title="API Deteksi Cyberbullying", version="1.0")

# Download NLTK data saat aplikasi start (penting untuk Cloud)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- 2. Load Model & Vectorizer ---
try:
    model = joblib.load('model_svm_cyberbullying.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    print("Model dan Vectorizer berhasil dimuat.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tfidf = None

# --- 3. Definisi Preprocessing (HARUS SAMA dengan saat Training) ---
# Stopwords (Gabungan Indo + Custom dari kode Anda sebelumnya)
indo_stopwords = set(stopwords.words('indonesian'))
custom_stopwords = {
    'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 'biar',
    'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si',
    'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 'jgn', 'sdh', 'aja', 'n',
    't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt', '&amp', 'yah'
}
final_stopwords = indo_stopwords.union(custom_stopwords)

def clean_text_indo(text):
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) # Hapus username
    text = re.sub(r'#', '', text)              # Hapus hashtag
    text = re.sub(r'\d+', '', text)            # Hapus angka
    text = re.sub(r'[^\w\s]', '', text)        # Hapus simbol
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)           # Hapus spasi ganda
    
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in final_stopwords]
    
    return " ".join(filtered)

# --- 4. Request Schema (Format data yang dikirim user) ---
class TextInput(BaseModel):
    text: str

# --- 5. Endpoint API ---

@app.get("/")
def home():
    return {"message": "API Cyberbullying Detection is Running!"}

@app.post("/predict")
def predict_sentiment(input_data: TextInput):
    if not model or not tfidf:
        return {"error": "Model belum dimuat dengan benar."}

    # Ambil teks dari input user
    raw_text = input_data.text
    
    # Lakukan preprocessing
    clean_text = clean_text_indo(raw_text)
    
    # Transformasi ke TF-IDF
    vectorized_text = tfidf.transform([clean_text]).toarray()
    
    # Prediksi
    prediction = model.predict(vectorized_text)[0]
    
    # Mapping hasil (Sesuaikan dengan Label Encoder Anda sebelumnya)
    # Contoh: Jika 0 = Non-Bullying, 1 = Bullying (Cek label_map Anda)
    # Asumsi umum:
    label_result = "Bullying" if prediction == 1 else "Non-Bullying"
    
    return {
        "text_input": raw_text,
        "text_cleaned": clean_text,
        "prediction_label": int(prediction),
        "result": label_result
    }