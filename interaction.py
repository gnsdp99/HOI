import numpy as np
import pandas as pd

X_train, X_test, y_train, y_test

def relu(x):
    return np.maximum(0, x)

def softmax(np_array):
    max_val = max(np_array)
    exp = np.exp(np_array - max_val)
    exp_sum = np.sum(exp)
    y = exp / exp_sum
    return y

def init_network():
    network = {}
    network['W1'] = np.random.rand(2, 3)
    network['b1'] = np.ones(3,)
    network['W2'] = np.random.rand(3, 2)
    network['b2'] = np.ones(2,)
    network['W3'] = np.random.rand(2, 1)
    network['b3'] = np.ones(1,)
    return network

def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.matmul(x, w1) + b1
    z1 = relu(a1)
    a2 = np.matnul(z1, w2) + b2
    z2 = relu(a2)
    a3 = np.matmul(z2, w3) + b3
    y = softmax(a3)
    return y

def cross_entropy_ohe(y, t): 
    if y.ndim == 1: 
        y = y.reshape(1, y.size) 
        t = t.reshape(1, t.size) 
    batch_size = y.shape[0] 
    loss = -np.sum(t * np.log(y + 1e-7)) / batch_size
    return loss

def numerical_gradient(f, x: np.array): 
    h = 1e-4 
    grad = np.zeros_like(x) 
    for idx in range(x.size):
        temp_val = x[idx] 
        x[idx] = temp_val + h 
        fxh1 = f(x)
        x[idx] = temp_val - h 
        fxh2 = f(x) 
        grad[idx] = (fxh1 - fxh2) / (2*h) 
        x[idx] = temp_val 
        return grad 
    
def gradient_descent(f, init_x: np.array, lr=0.01, step_num=100):
    x = init_x
    for _ in range(step_num): 
        grad = numerical_gradient(f, x) 
        x -= lr * grad 
    return x 

def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_ohe(y, t)
    
def accuracy(self, x, t):
        y = self.predict(x) 
        y = np.argmax(y, axis=1) 
        t = np.argmax(t, axis=1) 
        acc = np.sum(y == t) / float(x.shape[0]) 
        return acc
    
    