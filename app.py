import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================
# CACHE MODEL (IMPORTANT)
# ==============================

@st.cache_resource
def load_model():
    data = pd.read_csv("IMDB_dataset.csv")
    
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        words = text.split()
        words = [ps.stem(word) for word in words if word not in stop_words]
        return ' '.join(words)

    data['cleaned_review'] = data['review'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_review']).toarray()
    y = data['sentiment'].map({'positive':1, 'negative':0})

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, vectorizer, clean_text

model, vectorizer, clean_text = load_model()

# ==============================
# STREAMLIT UI
# ==============================

st.title("🎬 Movie Review Sentiment Analysis")

user_input = st.text_area("Enter a movie review:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    
    if prediction == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😡")