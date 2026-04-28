import numpy as np
from sklearn.datasets import fetch_openml

data = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
df = data.frame
df['sex'] = df['sex'].map({'male': 0, 'female': 1}).astype(int)
df = df[['pclass', 'sex', 'age', 'survived']].dropna()
X_raw  = df.drop('survived', axis=1)

#scaling 

class ScratchScaler():

    def __init__(self):
        self.means = None
        self.stds = None
    
    def fit_transform(self, x):
        data = x.copy()

        self.means = data.mean(axis=0)
        self.stds = np.std(data,axis=0) + 1e-8
        z_Score = (data-self.means)/self.stds
        return z_Score



scaler = ScratchScaler()
X = scaler.fit_transform(X_raw)
y = df['survived'].astype(float).values.reshape(-1,1)

def train_test_split(X,y, test_size=0.2, random_state=42, stratify=y):
    np.random.seed(random_state)
    # falt y, find the index, shuffle the index, calculte test_size_len, seprate train and test 
    y_flat = stratify.flatten() if hasattr(stratify, 'flatten') else np.array(stratify)
    idx1 = np.where(y_flat==1)[0]
    idx0 = np.where(y_flat==0)[0]

    def split_idx(indices):
        np.random.shuffle(indices)
        test_set_len = int(len(indices)*test_size)
        train = indices[test_set_len:]
        test = indices[:test_set_len]
        return train, test
    
    idx_train0, idx_test0 = split_idx(idx0)
    idx_train1, idx_test1 = split_idx(idx1)

    train_idx = np.concatenate([idx_train0, idx_train1])
    test_idx = np.concatenate([idx_test0, idx_test1])

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    if hasattr(X,'iloc'):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    else:
        X_train, X_test =  X[train_idx], X[test_idx]
    if hasattr(y,'iloc'):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        y_train, y_test = y[train_idx], y[test_idx]



    return  X_train, X_test, y_train, y_test





x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)
 
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

w1 = np.random.randn(3,4)*np.sqrt(2/3)
b1 = np.zeros((1,4))
w2= np.random.randn(4,1)*np.sqrt(2/3)
b2 = np.zeros((1,1))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forwardpass(x, w1, b1, w2, b2):
    z1 = np.dot(x,w1)+b1    #hidden input
    a1 = np.maximum(0, z1)  #relu + hidden output

    z2 = np.dot(a1,w2)+b2
    a2 = sigmoid(z2)

    return a2

y_pred = forwardpass(x_train, w1, b1, w2, b2)


#loss function  ( BCE & MSE )

#Binary cross entropy -1/x E(ylog(p)+(1-y)log(1-p) )
def bce(y,y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15 )
    return -np.mean((y*np.log(y_pred)+(1-y)*np.log(1-y_pred)))

initial_loss = bce(y_train, y_pred)
print(initial_loss)


# def mse(x,y):
#     return np.mean(np.sum((y-y_pred)**2))

# Optimization: Update the weights (W = W - learning_rate * gradient


# with scratch loss = 0.6931562656885067





def backwardpass(X, y, w1, b1, w2, b2):
    m = X.shape[0]

    # forward again (needed for gradients)
    z1 = np.dot(X, w1)+b1
    a1 = np.maximum(0,z1)

    z2 = np.dot(a1, w2)+b2
    a2 = sigmoid(z2)

    #  Backdrop

    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * (z1 > 0)

    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


# mlp = MLPClassifier(
#     hidden_layer_sizes=(4,),
#     activation='relu',
#     solver='adam',
#     random_state=42,
#     early_stopping=True,
#     validation_fraction=0.1
# )

# mlp.fit(x_train,y_train.ravel())
# y_pred = mlp.predict(x_test)

# print( mlp.loss_)
# with MLPClassifier = 0.6016274424484372

lr = 0.01
epochs = 1000

for i in range(epochs):
    # forward
    z1 = np.dot(x_train, w1) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    loss = bce(y_train, a2)

    # backward
    dW1, db1, dW2, db2 = backwardpass(x_train, y_train, w1, b1, w2, b2)

    # update
    w1 -= lr * dW1
    b1 -= lr * db1
    w2 -= lr * dW2
    b2 -= lr * db2

    if i % 100 == 0:
        print(f"Loss at epoch {i}: {loss}")