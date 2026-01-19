!pip install transformers torch datasets scikit-learn


from transformers import AutoTokenizer, AutoModel
import torch

# DistilBERT: Smaller, faster version of BERT
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print(f"Model loaded: {model_name}")
print(f"Vocabulary size: {tokenizer.vocab_size}")


import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer

# # Download NLTK resources
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# lemmatizer = WordNetLemmatizer()

# ===== STEP 1: Load Your Data =====
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ResumeDataset.csv') 

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Unique categories: {df['Category'].nunique()}")

# ===== STEP 2: Clean Resume Text =====
def clean_resume(text):
    clean_text = re.sub(r'http\S+', '', text)
    clean_text = re.sub(r'RT|cc', '', clean_text)
    clean_text = re.sub(r'@\S+', '', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    # Remove special chars but KEEP the words
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text.lower()
    # DON'T lemmatize - BERT needs raw text!

# Apply cleaning
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

# Encode labels
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['Category'])

print(f"\nCategory mapping:")
for idx, category in enumerate(le.classes_):
    print(f"{idx}: {category}")

# ===== STEP 3: Generate BERT Embeddings =====
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\nUsing device: {device}")

def get_bert_embedding(text, max_length=512):
    """
    Extract BERT embedding for a resume.
    Returns a 768-dimensional vector (the [CLS] token representation)
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    ).to(device)

    # Get embeddings (no gradients needed for inference)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract [CLS] token embedding (represents the whole sentence)
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return cls_embedding.squeeze()

# Generate embeddings for all resumes (this takes a few minutes)
print("\nGenerating BERT embeddings 
embeddings = []
for idx, resume in enumerate(tqdm(df['cleaned_resume'], desc="Embedding progress")):
    embedding = get_bert_embedding(resume)
    embeddings.append(embedding)

X = np.array(embeddings)
y = df['category_encoded'].values

print(f"Embeddings shape: {X.shape}")  # Should be (num_resumes, 768)

# ===== STEP 4: Train-Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# ===== STEP 5: Train Logistic Classifier =====
print("\nTraining Logistic Regression classifier on BERT embeddings...")
clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
clf.fit(X_train, y_train)


# ===== STEP 6: Evaluate =====
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ===== STEP 7: Save Models =====
pickle.dump(clf, open('clf_bert.pkl', 'wb'))
pickle.dump(le, open('encoder_bert.pkl', 'wb'))
pickle.dump(tokenizer, open('tokenizer_bert.pkl', 'wb'))
pickle.dump(model, open('model_bert.pkl', 'wb'))

print("\n✅ Models saved:")
print("- clf_bert.pkl (Classifier)")
print("- encoder_bert.pkl (Label Encoder)")
print("- tokenizer_bert.pkl (Tokenizer)")
print("- model_bert.pkl (BERT Model)")


import json

# Load the notebook
with open('/content/drive/MyDrive/Colab Notebooks/Resume_Screening_BERT.ipynb.ipynb', 'r') as f:
    notebook = json.load(f)

# Remove problematic metadata.widgets
if 'metadata' in notebook:
    if 'widgets' in notebook['metadata']:
        del notebook['metadata']['widgets']

# Save the cleaned notebook
with open('Resume_Screening_BERT_Training_CLEAN.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✅ Notebook cleaned! Download 'Resume_Screening_BERT_Training_CLEAN.ipynb'")

# Download the notebook as .py file directly
from google.colab import files
import subprocess

# Use nbconvert to convert to Python script
# Add check=True to raise an exception if the command fails
# Add capture_output=True and text=True to get stdout/stderr for debugging
nbconvert_result = subprocess.run(
    ['jupyter', 'nbconvert', '--to', 'script',
     '/content/drive/MyDrive/Colab Notebooks/Resume_Screening_BERT.ipynb.ipynb',
     '--output', 'Resume_Screening_BERT_Training.py'],
    capture_output=True, text=True, check=False # Set check=False initially to print output, then we can change to True
)

# Print nbconvert's output for debugging
print("nbconvert STDOUT:", nbconvert_result.stdout)
print("nbconvert STDERR:", nbconvert_result.stderr)

# Check the return code to see if nbconvert was successful
if nbconvert_result.returncode != 0:
    print("Error: jupyter nbconvert failed to create the Python script.")
else:
    print("jupyter nbconvert successful.")
    # Download it only if conversion was successful
    files.download('Resume_Screening_BERT_Training.py')
