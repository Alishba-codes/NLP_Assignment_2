import fitz  # PyMuPDF
import re
import nltk
import pandas as pd
import numpy as np
import plotly.express as px
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Q1(a): PDF Reading and Text Extraction ---
pdf_path = "book.pdf" 
doc = fitz.open(pdf_path)
full_text = "".join([page.get_text() for page in doc])

print(f"Total Number of Pages: {len(doc)}")
print("\n--- Sample Extracted Text ---")
print(full_text[:500])

# --- Q1(b): Text Preprocessing ---
text = full_text.lower()
# Mandatory Regex Tasks
text = re.sub(r'\d+', '', text)           # Remove numbers
text = re.sub(r'[^\w\s]', '', text)       # Remove special symbols
text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

words = word_tokenize(text)
stop_words = set(stopwords.words('english'))
valid_words = [w for w in words if w not in stop_words]

print(f"\nTotal Stop Words Found: {len(words) - len(valid_words)}")
print(f"Valid Words Count: {len(valid_words)}")

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
print("\nStemming (Sample):", [ps.stem(w) for w in valid_words[:10]])
print("Lemmatization (Sample):", [lemmatizer.lemmatize(w) for w in valid_words[:10]])

# --- Q1(c): Feature Extraction ---
# One Hot Encoding
sample_ohe = np.array(list(set(valid_words[:50]))).reshape(-1, 1)
ohe = OneHotEncoder(sparse_output=False)
ohe_df = pd.DataFrame(ohe.fit_transform(sample_ohe), 
                      columns=ohe.get_feature_names_out(), 
                      index=sample_ohe.flatten())
print("\nOne Hot Encoding Output (Sample):")
print(ohe_df.head())

# TF-IDF
vectorizer = TfidfVectorizer(max_features=50) 
tfidf_matrix = vectorizer.fit_transform([text])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
print("\nTF-IDF Output (Tabular):")
print(tfidf_df)

# --- Q1(d): TF-IDF Scatter Plot Using Plotly ---
fig = px.scatter(x=vectorizer.get_feature_names_out(), 
                 y=tfidf_matrix.toarray().flatten(),
                 labels={'x': 'Words', 'y': 'TF-IDF Score'},
                 title='TF-IDF Values Scatter Plot')
fig.show()