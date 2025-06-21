import streamlit as st
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# =============================
# Load Models & Tokenizers
# =============================

# Text LSTM model
model_lstm = load_model('models/text_lstm.h5')
tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))
le_text = pickle.load(open('models/label_encoder_text.pkl', 'rb'))

# Audio CNN model
model_cnn = load_model('models/audio_cnn.h5')
le_audio = pickle.load(open('models/label_encoder_audio.pkl', 'rb'))
scaler = pickle.load(open('models/audio_scaler.pkl', 'rb'))

# Fusion Model (e.g. Random Forest)
fusion_model = joblib.load('models/fusion_rf.pkl')
vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))

# =============================
# Feature Extraction Functions
# =============================

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_scaled = scaler.transform([mfcc_mean])
    return mfcc_scaled, mfcc_mean.reshape(40, 1, 1)

def prepare_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    return padded

# =============================
# Streamlit UI
# =============================

st.title("🎭 Emotion Detection from Audio & Text")

st.sidebar.header("📎 Input Data")
text_input = st.sidebar.text_area("Enter a text message")
audio_file = st.sidebar.file_uploader("Upload a WAV audio file", type=['wav'])

if st.sidebar.button("Predict Emotion"):

    if not text_input and not audio_file:
        st.warning("Please provide either text or audio.")
    
    else:
        # TEXT PREDICTION
        if text_input:
            X_text_dl = prepare_text(text_input)
            pred_text = model_lstm.predict(X_text_dl)
            emotion_text = le_text.inverse_transform([np.argmax(pred_text)])[0]
            st.subheader("📝 Text Emotion Prediction")
            st.success(f"Emotion: **{emotion_text}**")

        # AUDIO PREDICTION
        if audio_file:
            with open("temp.wav", "wb") as f:
                f.write(audio_file.read())
            mfcc_scaled, mfcc_reshaped = extract_audio_features("temp.wav")
            mfcc_reshaped = mfcc_reshaped[np.newaxis, :, :, np.newaxis]
            pred_audio = model_cnn.predict(mfcc_reshaped)
            emotion_audio = le_audio.inverse_transform([np.argmax(pred_audio)])[0]
            st.subheader("🎧 Audio Emotion Prediction")
            st.success(f"Emotion: **{emotion_audio}**")

        # FUSION
        if text_input and audio_file:
            tfidf_text = vectorizer.transform([text_input]).toarray()
            fusion_input = np.hstack((tfidf_text, mfcc_scaled))
            fusion_pred = fusion_model.predict(fusion_input)
            fusion_emotion = le_text.inverse_transform(fusion_pred)[0]
            st.subheader("🔀 Fusion Model Prediction")
            st.success(f"Emotion: **{fusion_emotion}**")
