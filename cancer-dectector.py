import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
df = pd.read_csv('data.csv')
df.drop(columns=['id'],inplace=True)
df['diagnosis'] = (df['diagnosis'] == 'B').astype(int)
x  = df.drop(columns=['diagnosis']).values
y = df['diagnosis'].values
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_hat = lr.predict(x_test)
report = classification_report(y_test,y_hat)
print('classification report: ',report)