
import streamlit as st
import pickle
import string
import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from nltk import word_tokenize

from nltk.corpus import stopwords
from torchvision import transforms


df=pd.read_csv('spam.csv', encoding="ISO-8859-1")
df.rename(columns = {'v2':'Message'}, inplace = True)
lm=WordNetLemmatizer()
sw=stopwords.words('english')
def transform_text(text):
    corpus=[]
    for i in df['Message']:
      t=i.lower()                           
      t=re.sub('[^A-Za-z0-9]',' ',t)        
      t=word_tokenize(t)                    
      t=[x for x in t if x not in sw]       
      t=[lm.lemmatize(x) for x in t]        
      t=" ".join(t)                         
      corpus.append(t)
    


tfidf= pickle.load(open('vectorize.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))

st.title("Email/SMS spam classifier")

input_sms= st.text_input("enter a message")

transformed_sms=transform_text(input_sms)
vector_input=tfidf.transform([transformed_sms])

result=model.predict(vector_input)[0]

if result== 1:
    st.header("spam")
else:
    st.header("Not Spam")

