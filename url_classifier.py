import numpy as np
import pandas as pd
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv') 
df = pd.DataFrame(df)
df = df.sample(n=10000)
col = ['label','url']
df = df[col]
#Deleting nulls
df = df[pd.notnull(df['url'])]
#more settings for our data manipulation
df.columns = ['label', 'url']
df['category_id'] = df['label'].factorize()[0]
category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'label']].values)

def get_tokens(inp):
    tokens_slash = str(inp.encode('utf-8')).split('/')
    all=[]
    for i in tokens_slash:
        tokens = str(i).split('-')
        tokens_dot = []
        for j in range(0,len(tokens)):
            temp = str(tokens[j]).split('.')
            tokens_dot = tokens_dot + temp
        all = all + tokens + tokens_dot
    all = list(set(all))
    if 'com' in all:
        all.remove('com')
    return all



vectorizer = TfidfVectorizer(tokenizer=get_tokens ,use_idf=True, smooth_idf=True, sublinear_tf=False)
features = vectorizer.fit_transform(df.url).toarray()

model = LogisticRegression(random_state=0)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, df.label, df.index, test_size=0.20, random_state=0)
model.fit(X_train, y_train)


with open("url_classifier.pkl", "wb") as file:
    pickle.dump(model, file)
