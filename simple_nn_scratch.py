import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

data = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
df = data.frame
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = df[['pclass', 'sex', 'age', 'survived']].dropna()
X_raw  = df.drop('survived', axis=1)

#scaling 
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
y = df['survived'].astype(float).values.reshape(-1,1)
 
# 1. Pick 3 features (must match the '3' in W1)
# Make sure they are scaled! Neural nets hate raw unscaled numbers.
# we need w1, b1, w2, b2 -> sigmoid -> relu 
# np.random.seed(42)


""" 
Matrix	    Shape	Role	                                Rule
Input (X)	(N, 3)	3 Features (Age, Fare, Sex)	            N = number of samples
W1	        (3, 4)	Bridge from 3 inputs 4 neurons	        (In, Out)
b1	        (1, 4)	One shift for each of the 4 neurons	    Matches W1 Out
W2	        (4, 1)	Bridge from 4 neurons 1 output	        (In, Out)
b2	        (1, 1)	One shift for the final prediction	    Matches W2 Out 
"""

w1 = np.random.randn(3,4)*0.01
b1 = np.zeros((1,4))
w2= np.random.randn(4,1)*0.01
b2 = np.zeros((1,1))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forwardpass(x, w1, b1, w2, b2):
    z1 = np.dot(x,w1)+b1    #hidden input
    a1 = np.maximum(0, z1)  #relu + hidden output

    z2 = np.dot(a1,w2)+b2
    a2 = sigmoid(z2)

    return a2

x=np.array([0.3,0.5,0.8])
y_pred = forwardpass(x, w1, b1, w2, b2)


#loss function  ( BCE & MSE )

#Binary cross entropy -1/x E(ylog(p)+(1-y)log(1-p) )
def bce(y,y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15 )
    return -np.mean((y*np.log(y_pred)+(1-y)*np.log(1-y_pred)))

initial_loss = bce(y, y_pred)

print(initial_loss)
# def mse(x,y):
#     return np.mean(np.sum((y-y_pred)**2))

# Optimization: Update the weights (W = W - learning_rate * gradient
