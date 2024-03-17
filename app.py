import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(Tekst):
    Tekst = Tekst.lower()
    Tekst = nltk.word_tokenize(Tekst)

    a = []
    for i in Tekst:
        if i.isalnum():
            a.append(i)

    Tekst = a[:]
    a.clear()

    for i in Tekst:
        if i not in stopwords.words('english') and i not in string.punctuation:
            a.append(i)

    Tekst = a[:]
    a.clear()

    for i in Tekst:
        a.append(ps.stem(i))

    return " ".join(a)

tfidfv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Klasifikimi i spam e-mail ")

input_sms = st.text_area("Shkruaj mesazhin tuaj ")

if st.button('Proceso'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidfv.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Legjitim")