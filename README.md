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

📈 Project Evolution & Improvements
v1.0 - Initial Baseline (Dec 2025)
Approach: TF-IDF Vectorization + Support Vector Machine (SVM)

Accuracy: 95% (test set)

Model Size: 181 MB (clf.pkl)

Features: Text file uploads only

Status: Working but undeployed (file size too large)

Challenge: Model file was too large for free cloud deployment.

v1.1 - Optimization Sprint (Dec 2025)
Change: Reduced TF-IDF max_features from 6,000 → 1,500

Accuracy: 98% (actually improved!)

Model Size: 18 MB (90% reduction)

Inference Speed: Instant

Insight: Vocabulary pruning removed noise without sacrificing performance

Key Learning: More features ≠ better accuracy. The top 1,500 words capture 99% of the signal.

v1.2 - Feature Expansion (Jan 2026)
Added: PDF file parsing using PyPDF2

Added: Robust text preprocessing with NLTK lemmatization

Accuracy: Maintained at 98%

Deployment: ✅ Live on Streamlit Cloud

Live URL: [(https://nlp-resume-screening-app-psdl4ecnsgvf3tzhgn2ced.streamlit.app/)]

Why PDF Support: Most job applications use PDFs. Raw binary text was being misclassified as "HR".

v2.0 (Experimental) - BERT Exploration (Jan 2026)
Approach: DistilBERT embeddings + Logistic Regression

Accuracy: 90.67%

Model Size: 268 MB

Inference Speed: Slower (GPU/CPU required for embeddings)

Branch: feature/bert-implementation

Why it Underperformed:

Dataset size (962 resumes) too small for transformers (typically need 10k+)

Task is keyword-based; TF-IDF is purpose-built for this

Trade-off: Contextual understanding vs. keyword precision

Outcome: Documented as research artifact. Demonstrates exploration of state-of-the-art techniques and rigorous A/B testing.

Key Metrics Comparison:
| Metric      | v1.0    | v1.1    | v1.2    | v2.0 (BERT) |
| ----------- | ------- | ------- | ------- | ----------- |
| Accuracy    | 95%     | 98%     | 98%     | 90.67%      |
| Model Size  | 181 MB  | 18 MB   | 18 MB   | 268 MB      |
| Inference   | Instant | Instant | Instant | Slow        |
| Deployable  | ❌      | ✅     | ✅      | ❌         |
| PDF Support | ❌      | ❌     | ✅      | ✅         |

Engineering Decisions
Feature Reduction (v1.0 → v1.1)

Analyzed feature importance; top 1,500 words sufficient

Result: 90% size reduction, 3% accuracy gain

PDF Parsing (v1.1 → v1.2)

Integrated PyPDF2 to extract text from PDFs

Solved silent classification failures (all PDFs were returning "HR")

BERT Experimentation (v2.0)

Explored modern transformer-based NLP

Learned that sophistication ≠ superior results

Maintained production-ready TF-IDF; documented BERT research

Production vs. Research
Production (main branch):

TF-IDF + SVM

98% accuracy

Lightweight & fast

Deployed on Streamlit Cloud

Research (feature/bert-implementation):

BERT + Logistic Regression

90.67% accuracy

Demonstrates exploration of advanced techniques

Documents decision-making process
