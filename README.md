# 📄 AI-Powered Resume Screening System

A Machine Learning-powered web application that automatically categorizes resumes into distinct job domains (e.g., Data Science, Java Developer, HR) using Natural Language Processing (NLP).

🔗 **Live Demo:** [Click Here to launch App](https://nlp-resume-screening-app-psdl4ecnsgvf3tzhgn2ced.streamlit.app/)

## 🛠️ Tech Stack
*   **Language:** Python
*   **Frontend:** Streamlit
*   **ML Libraries:** Scikit-Learn, NLTK, Pandas, NumPy
*   **Text Processing:** TF-IDF Vectorization, Regex Cleaning, Lemmatization
*   **Workflow:** Data Cleaning -> Feature Extraction -> SVM Classification

## 🚀 Key Features
*   **Multi-Format Support:** Extracts text from PDF, DOCX, and TXT files.
*   **Robust Cleaning:** Handles URLs, special characters, and hashtags via regex & lemmatization.
*   **High Accuracy:** Trained on a dataset of [Mention size if known, e.g., 900+] resumes.
*   **Real-time Prediction:** Instant classification using a pre-trained SVM model.

## 📂 Project Structure
├── app.py # Main Streamlit application
├── clf.pkl # Trained Classifier (SVM)
├── tfidf.pkl # TF-IDF Vectorizer
├── encoder.pkl # Label Encoder
├── requirements.txt # Dependencies
└── README.md # Documentation

## 🔧 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/RTE404/NLP-Resume-Screening-App.git
Install dependencies:

bash
pip install -r requirements.txt
Run the app:

bash
streamlit run app.py

