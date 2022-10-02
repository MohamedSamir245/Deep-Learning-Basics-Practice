import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("D:\Machine learning/nn_codebasics\L12-Gradient_Descent\insurance_data.csv")
df

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age','affordibility']],df[['bought_insurance']],test_size=0.2,random_state=25)
len(X_train)

X_train_scaled=X_train.copy()
X_train_scaled['age']=X_train_scaled['age']/100

X_test_scaled=X_test.copy()
X_test_scaled['age']=X_test_scaled['age']/100

model=keras.Sequential([
    keras.layers.Dense(1,input_shape=(2,),activation='sigmoid',kernel_initializer='ones',bias_initializer='zeros')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train_scaled,y_train,epochs=5000)


print(model.evaluate(X_test_scaled,y_test))
