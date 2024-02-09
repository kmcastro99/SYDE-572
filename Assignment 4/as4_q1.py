"""
SYDE 572 - Assignment 4

Author: Karla Castro

Question 1

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
np.random.seed(5)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100, convergence_threshold=1e-4):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.convergence_threshold = convergence_threshold
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Number of samples and number of features
        n_samples, n_features = X.shape
        self.bias = 0
        self.weights = np.zeros(n_features)

        loss_list = []
        errors_list = []

        for epoch in range(self.epochs):
            # Shuffle the data
            permutation = np.random.permutation(n_samples)
            X = X[permutation]
            y = y[permutation]

            # Forward pass
            Z = np.dot(X, self.weights) + self.bias  # Linear combination of features and parameters

            # Apply sigmoid activation function to obtain probabilities
            A = sigmoid(Z)  # Probabilities

            # Compute loss with clipping for numerical stability
            epsilon = 1e-15
            A_clipped = np.clip(A, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(A_clipped) + (1 - y) * np.log(1 - A_clipped))

            # Compute error
            y_pred_class = A > 0.5
            y_pred_class = np.array(y_pred_class, dtype='int64')
            errors = np.mean(np.abs(y_pred_class - y))

            # Compute gradients
            dz = A - y  # error of residuals
            dw = (1 / n_samples) * np.dot(X.T, dz)  # gradient of weights
            db = (1 / n_samples) * np.sum(dz)  # gradient of bias

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss_list.append(loss)
            errors_list.append(errors)

            # Check for convergence
            if epoch > 0 and np.abs(loss_list[epoch] - loss_list[epoch - 1]) < self.convergence_threshold:
                print(f'Convergence achieved at epoch {epoch}')
                break

            if epoch % (self.epochs / 10) == 0:
                print(f'Loss after iteration {epoch} is: {loss}')
                print(f'Error after iteration {epoch} is: {errors}')

        return self.weights, self.bias, loss_list, errors_list

    def predict(self, X):
        Z = np.dot(X, self.weights) + self.bias
        A = sigmoid(Z)
        y_pred_class = A > 0.5
        return np.array(y_pred_class, dtype='int64')

    def accuracy(self, X, y):
        y_pred_class = self.predict(X)
        acc = np.mean(y_pred_class == y)
        return acc * 100

# Load MNIST dataset using PyTorch
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=None)
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=None)

# Extract features and labels
X_train = mnist_train.data.numpy().reshape(-1, 28 * 28)
y_train = mnist_train.targets.numpy()

X_test = mnist_test.data.numpy().reshape(-1, 28 * 28)
y_test = mnist_test.targets.numpy()

# Keep only the first two classes
X_train = X_train[(y_train == 0) | (y_train == 1)]
y_train = y_train[(y_train == 0) | (y_train == 1)]

X_test = X_test[(y_test == 0) | (y_test == 1)]
y_test = y_test[(y_test == 0) | (y_test == 1)]

# Convert labels to integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initialize the model
model = LogisticRegression(learning_rate=0.01, epochs=500, convergence_threshold=1e-4)

print('Training data')
weights_train, bias_train, loss_train, errors_train = model.fit(X_train_pca, y_train)
# Accuracy
acc_train = model.accuracy(X_train_pca, y_train)
print(f'Accuracy for training data: {acc_train}')

print('Test data')
weights_test, bias_test, loss_test, errors_test = model.fit(X_test_pca, y_test)
acc_test = model.accuracy(X_test_pca, y_test)
print(f'Accuracy for test data: {acc_test}')

# Plot Model
def plot_model(loss, errors, type):
    last_epochs = np.arange(len(loss)) + 1
    x = last_epochs

    # Plot cost
    plt.plot(x[-5:], loss[-5:])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if type == 'train':
        plt.title('Loss vs Number of Epochs (Training Data)')
    else:
        plt.title('Loss vs Number of Epochs (Test Data)')
    plt.show()

    # Plot error
    plt.plot(x[-5:], errors[-5:])
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    if type == 'train':
        plt.title('Error vs Number of Epochs (Training Data)')
    else:
        plt.title('Error vs Number of Epochs (Test Data)')
    plt.show()

plot_model(loss_train, errors_train, 'train')
plot_model(loss_test, errors_test, 'test')


