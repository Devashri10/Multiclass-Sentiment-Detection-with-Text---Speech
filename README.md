# Multiclass Sentiment Detection from Text and Audio

This project aims to detect **emotions** (such as happy, sad, angry, etc.) from both **textual and audio inputs** using classical Machine Learning (ML), Deep Learning (DL), and a fusion of both modalities.

---

## Project Highlights

- **Text Emotion Detection** using:
  - TF-IDF + ML models (SVM, Logistic Regression, Random Forest, XGBoost)
  - LSTM-based deep learning on tokenized sequences

- **Audio Emotion Detection** using:
  - MFCC features + ML models
  - CNN-based deep learning for raw audio features

- **Fusion Model** combining text and audio features for improved prediction

---

## Dataset Used

- `train.txt`: Contains text samples with emotion labels.
- [TESS Dataset](https://tspace.library.utoronto.ca/handle/1807/24487](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess ): Audio recordings labeled by emotion in folders.(took this Audio Dataset From Kaggle)

---

## How to Use

1. Clone the repository or open the notebook in Google Colab.
2. Ensure you have required libraries installed (see below).
3. Upload `train.txt` and `TESS` dataset into your working directory.
4. Run all cells to train and evaluate models.
5. Optionally, use the prediction UI (Streamlit) for testing with custom input.

---

## Dependencies

```bash
pip install numpy pandas scikit-learn librosa xgboost matplotlib seaborn tensorflow
