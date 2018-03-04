import numpy as np
import numpy.linalg


# Hyper parameters
hidden_layer_size = 10


network = {
    'weights hidden': np.random.randn(hidden_layer_size, 1),
    'bias hidden': np.zeros(hidden_layer_size, 1),
    'weights out': np.random.randn(1, hidden_layer_size),
    'bias out': 0
}


def activation(z):
    return 1 / (1 + np.exp(-z))


def forward_pass(x, nn):
    """
    Calculate the forward pass of the Neural Network.

    Return the output, as well as the hidden layer's values.
    """
    z_1 = nn['weights hidden'] * x + nn['bias hidden']
    a_1 = activation(z_1)

    out = nn['weights out'] * a_1 + nn['bias out'], a_1

    return out, a_1


def loss(y, y_hat):
    return 1/(2 * len(y)) * np.linalg.norm(y - y_hat, ord=2)


def activation_derivative(a):
    return a * (1 - a)


def grad_loss(x, y, y_hat, hidden, model):
    """
    Calculate the gradient of loss for the weights and biases.

    x, y, y_hat are 1 x N row vectors.
    hidden is a N x H matrix where N is the number of observations and H is the hidden layer size.
    model is the model.
    """
    grad = dict()
    
    N = len(y)

    dJ_dy_hat = (1 / N) * (y - y_hat)
    dJ_dy_hat = dJ_dy_hat.T

    grad['bias out'] = dJ_dy_hat.sum()
    grad['weights out'] = np.matmul(hidden, dJ_dy_hat)
 
    hidden_prime = activation_derivative(hidden)

    dy_hat_db_hidden = model['weights out'] * hidden_prime
    grad['bias hidden'] =  np.matmul(dy_hat_db_hidden, dJ_dy_hat)

    dy_hat_dw_hidden = dy_hat_db_hidden * x
    grad['weights hidden'] = np.matmul(dy_hat_dw_hidden, dJ_dy_hat) 

    return grad

