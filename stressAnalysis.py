# stress_app.py

import pandas as pd
import numpy as np
import streamlit as st
import nltk
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("stress.csv")

# Preprocessing
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)

data["text"] = data["text"].apply(clean)

# Use existing labels (assumed to already be 'Stress' / 'No Stress')
data = data[["text", "label"]]

# Vectorization and model training
cv = CountVectorizer()
X = cv.fit_transform(data["text"])
y = data["label"]
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

model = BernoulliNB()
model.fit(xtrain, ytrain)

# Streamlit UI
st.title("Stress Detection from Text")
user_input = st.text_area("Enter text to analyze:", "")

if st.button("Predict"):
    cleaned_input = clean(user_input)
    vector = cv.transform([cleaned_input]).toarray()
    prediction = model.predict(vector)
    st.write(f"**Prediction:** {prediction[0]}")
