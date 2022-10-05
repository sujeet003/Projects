import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import pickle
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("train.csv")



df.fillna(df.median(),inplace=True)

x=df.drop(['song_popularity'],axis=1)
y = df["song_popularity"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)

lr=LogisticRegression()
lr.fit(x,y.values.ravel())

# dumping the model in pickle file
with open('model.pkl', 'wb') as files:
    pickle.dump(lr, files)
