from random import random
from math import exp


# parameters
train_size = 1000

inputs_size = 3
layer1_size = 20
layer2_size = 20
outputs_size = 1


# The function we are trying to estimate
def func(a, b, c):
    return a + b + c

def random_matrix(nrow, ncol):
    return [[random()*2 - 1 for _ in range(ncol)] for _ in range(nrow)]

def activation(x):
    return 1.0 / (1 + exp(-x))


hidden_weights1 = random_matrix(inputs_size, layer1_size)
hidden_weights2 = random_matrix(layer1_size, layer2_size)
output_weights = random_matrix(layer2_size, outputs_size)


def dot_prod(u, v):
    return sum([a * b for a, b in zip(u, v)])

def matrix_prod(A, B):
    return [[dot_prod(a, b) for a in zip(*A)] for b in B]


def forward_pass(inputs):
    x = inputs

    # First layer
    x = matrix_prod(hidden_weights1, x)
    x = [[activation(a) for a in row] for row in x]

    # Second layer
    x = matrix_prod(hidden_weights2, x)
    x = [[activation(a) for a in row] for row in x]

    # Output layer
    x = matrix_prod(output_weights, x)

    return [r for row in x for r in row]


def cost(y, y_hat):
    return sum([(a - b)**2 for a, b in zip(y, y_hat)])/(2*len(y))


# Training data
training = random_matrix(train_size, inputs_size)
target = [func(a, b, c) for a, b, c in training]

prediction = forward_pass(training)
