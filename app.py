import streamlit as st
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer

# --- Download NLTK Resources ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Load Models ---
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# --- Initialize Lemmatizer ---
lemmatizer = WordNetLemmatizer()

def clean_resume(text):
    clean_text = re.sub(r'http\S+', '', text)
    clean_text = re.sub(r'RT|cc', '', clean_text)
    clean_text = re.sub(r'@\S+', '', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    clean_text = re.sub(r'[' + re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""") + r']', ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    clean_text = clean_text.lower()
    words = clean_text.split()

    # Lemmatization
    lemmatized_words = []
    for word in words:
        if len(word) > 2:
            word = lemmatizer.lemmatize(word, pos='v')
            word = lemmatizer.lemmatize(word, pos='n')
            lemmatized_words.append(word)

    return " ".join(lemmatized_words)

def main():
    st.title("Resume Screening App")
    st.write("Upload your resume to see the predicted job category")

    uploaded_file = st.file_uploader("Upload Resume", type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_text = ""
            
            # --- PDF HANDLING ---
            if uploaded_file.name.endswith('.pdf'):
                # We need PyPDF2 for this
                import PyPDF2
                
                # Create a PDF Reader object
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                
                # Loop through all pages and extract text
                for page in pdf_reader.pages:
                    resume_text += page.extract_text() + " "
                    
            # --- TXT HANDLING ---
            else:
                resume_bytes = uploaded_file.read()
                try:
                    resume_text = resume_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    resume_text = resume_bytes.decode('latin-1')

            # --- PREDICTION LOGIC ---
            cleaned_resume = clean_resume(resume_text)
            input_features = tfidf.transform([cleaned_resume])
            input_features = input_features.toarray() # Convert to dense
            prediction_id = clf.predict(input_features)[0]

            # Map category ID to name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }
            
            category_name = category_mapping.get(prediction_id, "Unknown")
            st.write("Predicted Category:", category_name)

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
