import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk

from nltk.stem.porter import PorterStemmer
import pandas as pd
import re
ps = PorterStemmer()
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download('stopwords')

lm = WordNetLemmatizer()
sw = stopwords.words('english')
print(sw)

df = pd.read_csv('spam.csv', encoding="ISO-8859-1")
df.rename(columns={'v1':'category'}, inplace=True)
df.rename(columns={'v2':'Message'}, inplace=True)
df.drop_duplicates(inplace=True)

df1 = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)

corpus = []
for i in df1['Message']:
    t = i.lower()
    t = re.sub('[^A-Za-z0-9]',' ',t)
    t = word_tokenize(t)
    t = [x for x in t if x not in sw]
    t = [lm.lemmatize(x) for x in t]
    t = " ".join(t)
    corpus.append(t)

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
sm = cv.fit_transform(corpus).toarray()
print(sm.shape)

x = sm
y = df1['category']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    a = cv.transform([transformed_sms])
    
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(x_train, y_train)
    result = model.predict(a)

    if result == ['spam']:
        st.header("Spam")
    else:
        st.header("Not Spam")
