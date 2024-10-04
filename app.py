import pandas as pd 
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

introduction = """

### About This Project:

This is a minor Natural Language Processing project using NLTK, Streamlit, Pandas and Scikit-learn 
libraries to classify a given movie review as positive or negative. The model has been trained on 
IMDB dataset of Kaggle which contains 50000 movie reviews classified as positive or negative.

"""

def page():
    st.title("Movie Reviews Sentiment Analysis")
    st.markdown(introduction)

page()

model = pk.load(open('model.pkl','rb'))
scaler = pk.load(open('scaler.pkl','rb'))
review = st.text_input('Enter Movie Review')

if st.button('Predict'):
    review_scale = scaler.transform([review]).toarray()
    result = model.predict(review_scale)
    if result[0] == 0:
        st.write('Negative Review')
    else:
        st.write('Positive Review')

