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

lm=WordNetLemmatizer()
sw=stopwords.words('english')
print(sw)


df=pd.read_csv('spam.csv', encoding="ISO-8859-1")
df.head()

df.rename(columns = {'v1':'category'}, inplace = True)

df.rename(columns = {'v2':'Message'}, inplace = True)
df.head()

df.duplicated().sum()

df.drop_duplicates()
df.head()

df1=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
df1.head()

df1['category'].value_counts()



df['Message'].head()

corpus=[]
for i in df['Message']:
  t=i.lower()                           
  t=re.sub('[^A-Za-z0-9]',' ',t)        
  t=word_tokenize(t)                    
  t=[x for x in t if x not in sw]      
  t=[lm.lemmatize(x) for x in t]        
  t=" ".join(t)                         
  corpus.append(t)

print(corpus)

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()
sm=cv.fit_transform(corpus).toarray()
print(sm.shape)

print(cv.get_feature_names())

x=sm
y=df['category']
print(type(x))
print(type(y))

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

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf = pickle.load(open('vectorize.pkl','rb'))

model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

   
    transformed_sms = transform_text(input_sms)
   
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Normalizer
    a = np.array([transformed_sms])
    sc = StandardScaler(with_mean=False)    
    a = tfidf.fit_transform(a)

    # a = sc.fit(a.reshape(-1,1))
    # vector_input = Normalizer().fit([transformed_sms])
    # vector_input = tfidf.transform([transformed_sms])
    print((a))
    
   
    from sklearn.naive_bayes import GaussianNB,MultinomialNB
    from sklearn.model_selection import train_test_split
    # result=model.fit(x_train,y_train)
    # result = model.predict(a)

   
    # if result == 1:
    #     st.header("Spam")
    # else:
    #     st.header("Not Spam")