import matplotlib.pyplot as plt

import numpy as np

"""
A simple linear model trained via backpropogation.
"""

import random 

linear_model = {
    'slope': 0,
    'y-intercept': 0
}


def forward_pass(model, x):
    return model['slope'] * x + model['y-intercept']


def loss(y, y_hat):
    return 1 / (2 * len(y)) * sum((y - y_hat)**2)


def generate_data(num_examples):
    """
    Generate some noisy data.
    """
    m = np.random.random_sample() + 1
    b = 2 * np.random.random_sample() - 1

    x = np.random.random_sample(num_examples) * 10
    x = x + 0.1 * np.random.randn(num_examples)

    y = m * x + b + np.random.randn(num_examples)

    return x, y


def grad_loss(x, y, y_hat):
    N = len(x)
    d_slope = 1 / N * sum((y - y_hat) * (-x))
    d_intercept = 1 / N * sum((y - y_hat) * -1)

    return d_slope, d_intercept


learning_rate = 0.01

x, y = generate_data(50)

plt.ion()

num_epochs = 20

losses = []

for epoch in range(num_epochs):

    # Forward pass 
    y_hat = forward_pass(linear_model, x)

    # Calculate Loss
    J = loss(y, y_hat)

    losses.append(J)

    print('Epoch: {} Loss: {}'.format(epoch, J))

    d_m, d_b = grad_loss(x, y, y_hat)

    # Update model
    linear_model['slope'] = linear_model['slope'] - learning_rate * d_m
    linear_model['y-intercept'] = linear_model['y-intercept'] - learning_rate * d_b

    # Plot this stage
    x_line = np.array([-1, 11])
    y_line = forward_pass(linear_model, x_line)

    plt.clf()
    plt.subplot(2, 1, 1)
    plt.axis([-1, 11, min(y) - 1, max(y) - 1])
    plt.plot(x, y, 'b.', alpha = 0.25)
    plt.plot(x_line, y_line, 'k--')

    plt.subplot(2, 1, 2)
    plt.axis([0, num_epochs, 0, losses[0]])
    plt.plot(range(len(losses)), losses,'g')
    plt.plot(range(len(losses)), losses,'g.')

    plt.pause(0.05)
    
input('Press Enter to quit.')  # Pause before closing
