import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return (x > 0).astype(float)

def softmax(x):
    """Computes the softmax of a 1D array or 2D array (rows as samples)."""
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x)) # For numerical stability
        return e_x / e_x.sum()
    else:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

#----Special Func----#

def neuro_spike(x, sharpness=5.0):
    return x * np.exp(-sharpness * (x - 1)**2)

def adaptive_pulse(x, beta=1.5):
    return np.tanh(x) / (1 + beta * np.abs(x))

def neuro_softmax(x, gamma=2.0):
    x = np.array(x)
    x_stable = x - np.max(x)
    exp_x = np.exp(gamma * x_stable)
    return exp_x / np.sum(exp_x)

