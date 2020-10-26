#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, render_template

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

import numpy as np
import pandas as pd

import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import dill


# In[2]:


# !pip install flask-ngrok
from flask_ngrok import run_with_ngrok


# In[3]:


nltk.download("stopwords")
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download("punkt")

model = keras.models.load_model("C:/Users/ashra/Documents/Enhance IT/NLP_group/")

    
with open("C:/Users/ashra/Documents/Enhance IT/NLP_group/token.pickle", "rb") as f:
    
    
  token = dill.load(f)

def normalizer(tweet):
    no_urls = re.sub(r"http\S+", " " ,tweet)
    only_letters = re.sub("[^a-zA-Z]", " ",no_urls)
    tokens = nltk.word_tokenize(only_letters)[2 :]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    #lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return filtered_result

#make a prediction on input text
def predict_on_text(text): 
  test_text = np.array([text])
  test_df =  pd.DataFrame(test_text, columns = ['text'])

  test_df['normalized_tweet'] = test_df.text.apply(normalizer)
  X = test_df["normalized_tweet"].astype(str)

  df = token.texts_to_sequences(X)

  print(df)

  df = tf.keras.preprocessing.sequence.pad_sequences(df, maxlen=100)

  print(df)

  prediction = np.round(model.predict(df)[0][0]-.45)

  if prediction:

    return "This is about Sports."

  else:

    return "This is not about Sports."


# In[7]:


app = Flask(__name__, template_folder= "C:/Users/ashra/Documents/Enhance IT/NLP_group/Templates")


# In[8]:


run_with_ngrok(app)   # Starts ngrok when the app is run


# In[9]:


@app.route("/",  methods=["POST", "GET"])
def home():

    message = ''
    if request.method == 'POST':
        text = request.form.get('test_text')  # access the data inside 
        message = predict_on_text(text)
    return render_template("home.html", message = message)

@app.route("/predict",  methods=["POST", "GET"])
def chat():
    return predict_on_text(request.json)


# In[10]:


# app.run(host='0.0.0.0', debug=False)


# In[ ]:
if __name__=="__main":
  app.run(debug=True)

#app.run()


# In[ ]:




