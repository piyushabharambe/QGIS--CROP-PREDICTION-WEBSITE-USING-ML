#!/usr/bin/env python
# coding: utf-8
from joblib import dump, load


import plotly.offline as pyo

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS
from PIL import Image

import plotly.graph_objs as go
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns = None


df=pd.read_csv('cpdata.csv')



from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(['df.Season'])
df['Season']=le.fit_transform(df['Season'])


le1=preprocessing.LabelEncoder()
le1.fit(['df.T_ype_of_soil'])
df['T_ype_of_soil']=le1.fit_transform(df['T_ype_of_soil'])




from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

target = ['Crop']
features = ['Season','Avg_Temperature','Relative_Humidity', 'Avg_Annual_Rainfall', 'T_ype_of_soil']

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
models = []
models.append(('LogisticRegression', LogisticRegression(random_state=0)))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier(random_state=0)))
models.append(('GradientBoostingClassifier', GradientBoostingClassifier(random_state=0)))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier(random_state=0)))

model_name = []
accuracy = []

for name, model in models: 
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    model_name.append(name)
    accuracy.append(metrics.accuracy_score(y_test,y_pred))
    #print(name, metrics.accuracy_score(y_test,y_pred))




model=RandomForestClassifier(random_state=0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn import metrics
import joblib
import pickle

#print(metrics.classification_report(y_test,y_pred))


Season=3
Avg_Temperature=311.15
Relative_Humidity=49
Avg_Annual_Rainfall=381
T_ype_of_soil=14
sample=[Season,Avg_Temperature,Relative_Humidity,Avg_Annual_Rainfall,T_ype_of_soil]
single_sample=np.array(sample).reshape(1,-1)
pred=model.predict(single_sample)
le_soil_mapping=dict(zip(le.classes_,le.transform(le.classes_)))
le_season_mapping=dict(zip(le1.classes_,le1.transform(le1.classes_)))

print(le_soil_mapping)
print(le_season_mapping)
print(pred.item().title())
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))
print(model.predict(single_sample))













