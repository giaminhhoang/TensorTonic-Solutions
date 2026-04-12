import numpy as np

def sigmoid(x):
    return 1 / (1.0 + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x)*(1 - sigmoid(x))

def forward(w1, w2, w3, x):
    z1 = w1*x
    a1 = sigmoid(z1)
    z2 = w2*a1
    a2 = sigmoid(z2)
    z3 = w3*a2
    y = sigmoid(z3)
    return z1, z2, z3, y

def chain_rule_3layer(w1, w2, w3, x):
    """
    Returns: dict with 'factors' (list of 6 floats), 'analytical_gradient' (float), 'numerical_gradient' (float)
    """
    z1, z2, z3, y = forward(w1, w2, w3, x)
    w1_grad = sigmoid_grad(z3)*w3*sigmoid_grad(z2)*w2*sigmoid_grad(z1)*x

    h = 1e-5
    _, _, _, yplus = forward(w1+h, w2, w3, x)
    _, _, _, yneg = forward(w1-h, w2, w3, x)
    w1_grad_num = (yplus - yneg)/2/h
    return {"factors": [sigmoid_grad(z3),w3,sigmoid_grad(z2),w2,sigmoid_grad(z1),x], "analytical_gradient": w1_grad, "numerical_gradient": w1_grad_num}