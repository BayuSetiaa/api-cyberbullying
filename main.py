from fastapi import FastAPI
from fastapi.responses import HTMLResponse # <--- INI YANG KURANG TADI
from pydantic import BaseModel
import joblib
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- KONFIGURASI NLTK KHUSUS CLOUD ---
# Kita download punkt_tab juga karena versi NLTK terbaru membutuhkannya
nltk.download('punkt_tab') 

# Buat folder lokal untuk menyimpan data NLTK
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

# Beritahu NLTK untuk mencari data di folder ini
nltk.data.path.append(nltk_data_dir)

# Fungsi helper untuk download aman
def download_nltk_data(package_name):
    try:
        nltk.data.find(f'tokenizers/{package_name}', paths=[nltk_data_dir])
    except LookupError:
        try:
            nltk.data.find(f'corpora/{package_name}', paths=[nltk_data_dir])
        except LookupError:
            print(f"Downloading {package_name}...")
            nltk.download(package_name, download_dir=nltk_data_dir)

# Download resource yang dibutuhkan
download_nltk_data('punkt')
download_nltk_data('stopwords')
# ------------------------------------

# --- 1. Inisialisasi App ---
app = FastAPI(title="API Deteksi Cyberbullying", version="1.0")

# --- 2. Load Model & Vectorizer ---
try:
    model = joblib.load('model_svm_cyberbullying.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    print("Model dan Vectorizer berhasil dimuat.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tfidf = None

# --- 3. Definisi Preprocessing ---
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
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) 
    text = re.sub(r'#', '', text)              
    text = re.sub(r'\d+', '', text)            
    text = re.sub(r'[^\w\s]', '', text)        
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)           
    
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in final_stopwords]
    
    return " ".join(filtered)

class TextInput(BaseModel):
    text: str

# --- 4. Endpoint ---

@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("index.html", "r", encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: File index.html tidak ditemukan. Pastikan file ada di folder yang sama.</h1>"

@app.post("/predict")
def predict_sentiment(input_data: TextInput):
    if not model or not tfidf:
        return {"error": "Model belum dimuat."}
    
    raw_text = input_data.text
    clean_text = clean_text_indo(raw_text)
    vectorized_text = tfidf.transform([clean_text]).toarray()
    prediction = model.predict(vectorized_text)[0]
    
    label_result = "Bullying" if prediction == 1 else "Non-Bullying"
    
    return {
        "text_input": raw_text,
        "prediction_label": int(prediction),
        "result": label_result
    }