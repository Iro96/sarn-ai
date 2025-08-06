import numpy as np

def neuro_spike(x, sharpness=5.0):
    return x * np.exp(-sharpness * (x - 1)**2)

def adaptive_pulse(x, beta=1.5):
    return np.tanh(x) / (1 + beta * np.abs(x))

def neuro_softmax(x, gamma=2.0):
    x = np.array(x)
    x_stable = x - np.max(x)
    exp_x = np.exp(gamma * x_stable)
    return exp_x / np.sum(exp_x)
