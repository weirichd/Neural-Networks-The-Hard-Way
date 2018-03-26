import numpy as np
import matplotlib.pyplot as plt

# Hyper parameters
hidden_layer_size = 100


neural_network = {
    'weights hidden': np.random.randn(hidden_layer_size, 1),
    'bias hidden': np.zeros((hidden_layer_size, 1)),
    'weights out': np.random.randn(1, hidden_layer_size),
    'bias out': 0.0
}


def activation(z):
    return 1 / (1 + np.exp(-z))


def forward_pass(x, model):
    """
    Calculate the forward pass of the Neural Network.

    Return the output, as well as the hidden layer's values.
    (We return the hidden layer's values here because it will be useful
    when calculating the gradient).
    """
    z_1 = np.matmul(model['weights hidden'], x) + model['bias hidden']
    a_1 = activation(z_1)

    out = np.matmul(model['weights out'], a_1) + model['bias out']

    return out, a_1


def loss(y, y_hat):
    J = 1/(2 * y.shape[1]) * ((y - y_hat)**2)
    return J.sum()


def generate_data(num_examples):
    """
    Generate some noisy data.
    """

    x = np.random.random_sample((1, num_examples)) * np.pi * 4 - np.pi * 2
    x = x + 0.1 * np.random.randn(*x.shape)

    y = 0.1 * x + np.cos(x) + 0.25 * np.random.randn(num_examples)

    return x, y


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
    
    N = y.shape[1]

    dJ_dy_hat = -(1 / N) * (y - y_hat)

    grad['bias out'] = dJ_dy_hat.sum()
    grad['weights out'] = np.matmul(dJ_dy_hat, hidden.T)
 
    hidden_prime = activation_derivative(hidden)

    dy_hat_db_hidden = model['weights out'].T * hidden_prime
    grad['bias hidden'] = np.matmul(dy_hat_db_hidden, dJ_dy_hat.T)

    dy_hat_dw_hidden = dy_hat_db_hidden * x
    grad['weights hidden'] = np.matmul(dy_hat_dw_hidden, dJ_dy_hat.T)

    return grad


# Train the model

learning_rate = 0.01

x, y = generate_data(200)

plt.ion()

num_epochs = 10000

losses = []

for epoch in range(num_epochs):

    # Forward pass 
    y_hat, hidden_values = forward_pass(x, neural_network)

    # Calculate Loss
    J = loss(y, y_hat)

    losses.append(J)

    gradient = grad_loss(x, y, y_hat, hidden_values, neural_network)

    # Update model
    neural_network['weights out'] = neural_network['weights out'] - learning_rate * gradient['weights out']
    neural_network['weights hidden'] = neural_network['weights hidden'] - learning_rate * gradient['weights hidden']
    neural_network['bias out'] = neural_network['bias out'] - learning_rate * gradient['bias out']
    neural_network['bias hidden'] = neural_network['bias hidden'] - learning_rate * gradient['bias hidden']

    # Plot this stage
    if epoch % 100 == 0 and epoch > 0:
        print('Epoch: {} Loss: {}'.format(epoch, J))

        plt.clf()
        plt.subplot(2, 1, 1)
        plt.axis([x.min() - 1, x.max() + 1, y.min() - 1, y.max() + 1])
        plt.plot(x, y, 'b.', alpha=0.25)
        plt.plot(x, y_hat, 'k.')

        losses_to_plot = losses[50::50]

        plt.subplot(2, 1, 2)
        plt.axis([0, num_epochs / 50, 0, losses_to_plot[0]])
        plt.plot(range(len(losses_to_plot)), losses_to_plot,'g')
        plt.plot(range(len(losses_to_plot)), losses_to_plot,'g.')

        plt.pause(0.02)


input('Press Enter to quit.')  # Pause before closing
