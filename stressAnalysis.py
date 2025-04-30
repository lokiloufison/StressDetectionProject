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
from sklearn.metrics import accuracy_score, classification_report
nltk.data.path.append("nltk_data")

# Load dataset
data = pd.read_csv("stress.csv")

# Check data distribution
st.write("### Distribution of Labels:")
label_counts = data['label'].value_counts()
st.write(label_counts)

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

# Evaluate model accuracy
st.write("### Model Accuracy on Test Set:")
predictions = model.predict(xtest)
st.write("Accuracy:", accuracy_score(ytest, predictions))

st.write("### Classification Report:")
st.write(classification_report(ytest, predictions))

# Streamlit UI
st.title("Stress Detection from Text")
user_input = st.text_area("Enter text to analyze:", "")

if st.button("Predict"):
    cleaned_input = clean(user_input)
    vector = cv.transform([cleaned_input]).toarray()
    prediction = model.predict(vector)
    
    # Debugging
    st.write("### Debugging Information:")
    st.write("Cleaned Input:", cleaned_input)
    st.write("Vectorized Input:", vector)
    st.write("Raw Prediction:", prediction)
    
    # Map 0 and 1 to 'No Stress' and 'Stress'
    prediction_label = "Stress" if prediction[0] == 1 else "No Stress"
    st.write(f"**Prediction:** {prediction_label}")
