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

# Create sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Create logistic regression class
class LogisticRegression():
    
    def __init__(self, learning_rate=0.1, epochs =100, convergence_threshold=1e-5):
        self.lerning_rate = learning_rate
        self.epochs = epochs
        self.convergence_threshold = convergence_threshold
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Number of samples and number of features
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        loss_list = []
        errors_list = []


        for epoch in range(self.epochs):

            # Forward pass
            Z = np.dot(X, self.weights) + self.bias # Linear combination of features and parameters

            # Apply sigmoid activation function to obtain probabilities
            A = sigmoid(Z) # Probabilities

            # Compute loss 
            loss = np.sum(y * np.log(A + 1e-15) + (1 - y) * np.log(1 - A+ 1e-15)) * (-(1 / n_samples))

            # Compute error
            y_pred_class = A > 0.5
            y_pred_class = np.array(y_pred_class, dtype= 'int64')
            errors = np.sum(np.abs(y_pred_class - y)) / y.shape[0]

            # Compute gradients
            dz = A - y # error of residuals
            dw = (1 / n_samples) * np.dot(X.T, dz) # gradient of weights
            db = (1 / n_samples) * np.sum(dz) # gradient of bias
            
            # Update parameters
            self.weights -= self.lerning_rate * dw
            self.bias -= self.lerning_rate * db

            loss_list.append(loss)
            errors_list.append(errors)

            # Check for convergence
            if epoch > 0 and np.abs(loss_list[epoch] - loss_list[epoch - 1]) < self.convergence_threshold:
                print(f'Convergence achieved at epoch {epoch}')
                break

            if (epoch %(self.epochs/100) == 0):
                print(f'Loss after iteration {epoch} is: {loss}')
                print(f'Error after iteration {epoch} is: {errors}')

        return self.weights, self.bias, loss_list, errors_list
    
    def accuracy(self, X, y, w, b):
        Z = np.dot(X, w) + b
        A = sigmoid(Z)
        y_pred_class = A > 0.5
        y_pred_class = np.array(y_pred_class, dtype= 'int64')
        acc = (1 - np.sum(np.abs(y_pred_class - y)) / y.shape[0]) * 100
        return acc


root = 'data'

# Load data class 0
dataset_0 = datasets.MNIST(root=root, train=True, download=True)
subset_idx = torch.isin(dataset_0.targets, torch.as_tensor(0))
train_mnist_0_raw = dataset_0.data[subset_idx].numpy()
train_mnist_0_labels = dataset_0.targets[subset_idx].numpy()
# Flatten images
train_mnist_0_flat = train_mnist_0_raw.reshape(-1, 28*28)
#--------------------
# Load data class 1
dataset_1 = datasets.MNIST(root=root, train=True, download=True)
subset_idx = torch.isin(dataset_1.targets, torch.as_tensor(1))
train_mnist_1_raw = dataset_1.data[subset_idx].numpy()
train_mnist_1_labels = dataset_1.targets[subset_idx].numpy()
# Flatten images
train_mnist_1_flat = train_mnist_1_raw.reshape(-1, 28*28)

# Perform PCA with 2 components
pca = PCA(n_components=2)
train_pca = pca.fit_transform(np.concatenate([train_mnist_0_flat, train_mnist_1_flat])).astype(np.float32)
np.random.shuffle(train_pca)
# Obtain datasets
train_mnist_0 = train_pca[:len(train_mnist_0_raw)]

train_mnist_1 = train_pca[len(train_mnist_1_raw):-1]

np.random.shuffle(train_mnist_0)
np.random.shuffle(train_mnist_1)

X_train = np.vstack((train_mnist_0, train_mnist_1))
y_train = np.hstack((train_mnist_0_labels, train_mnist_1_labels))

# Initialize the model
model = LogisticRegression(learning_rate=0.01, epochs=100)

print('Training data')
weigths_train, bias_train, loss_train, errors_train = model.fit(X_train, y_train) # Training data
# Accuracy
acc_train = model.accuracy(X_train, y_train, weigths_train, bias_train) # Training data
print(f'Accuracy for training data: {acc_train}')

# print('Test data')
# weigths_test, bias_test, loss_test, errors_test = model.fit(X_test, y_test) # Test data
# last_epochs = range(len(errors_train)-5, len(errors_train))
# acc_test = model.accuracy(X_test, y_test, weigths_test, bias_test) # Test data
# print(f'Accuracy for test data: {acc_test}')

# Plot Model

def plot_model(loss, errors, type):
    x = range(len(errors)-5, len(errors)) # last 5 epochs
    num_last_epochs = len(x)

    # Plot cost
    plt.plot(x, loss[-num_last_epochs:])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if type == 'train':
        plt.title('Loss vs Number of Epochs (Training Data)')
    else:
        plt.title('Loss vs Number of Epochs (Test Data)')
    plt.show()

    # Plot error
    plt.plot(x, errors[-num_last_epochs:])
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    if type == 'train':
        plt.title('Error vs Number of Epochs (Training Data)')
    else:
        plt.title('Error vs Number of Epochs (Test Data)')
    plt.show()

plot_model(loss_train, errors_train, 'train')
# plot_model(loss_test, errors_test, 'test')





