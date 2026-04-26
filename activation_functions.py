import numpy as np


def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/ 1+np.exp(-x)

def leakyrelu(x, alpha=0.01):
    return np.maximum(x*alpha, x)
 

# for 2d always keep keepdims and axis
def softmax(x, axis=-1): #( e^(x-X_max)-/E(e^(x-X_max)) 
    exps = np.exp(x - np.max(x, keepdims=True, axis=axis))
    return exps/np.sum(exps, keepdims=True, axis=axis)


x = np.linspace(0,10,20).reshape(10,2)

print(relu(x))
print(sigmoid(x))
print(leakyrelu(x, alpha=0.01))
print(softmax(x, axis=1))