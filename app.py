import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

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

