import numpy as np
import pandas as pd

o_mid = 0 # object's mid point
p_mid = 0 # person's mid point
k_h, k_rh, k_lh, k_rf, k_lf = keypoints # keypoint of head, right hand, left hand, right foot, left foot

features = [o_mid - p_mid, o_mid - k_h, o_mid - k_rh, o_mid - k_lh, o_mid - k_rf, o_mid - k_lf] # o_mid로부터 각 좌표까지의 거리
label = ['non-interact', 'interact']

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

