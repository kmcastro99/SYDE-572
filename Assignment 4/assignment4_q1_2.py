import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
from torchvision import datasets

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=100, convergence_threshold=1e-5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.convergence_threshold = convergence_threshold
        self.weights = None
        self.train_errors = []
        self.test_errors = []
        self.train_losses = []
        self.test_losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def add_bias_term(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def compute_loss(self, y_true, y_pred):
        m = len(y_true)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def gradient_descent(self, X, y, X_test=None, y_test=None):
        m, n = X.shape
        self.weights = np.zeros((n + 1, 1))  # Add 1 for the bias term

        X_with_bias = self.add_bias_term(X)
        
        if X_test is not None:
            X_test_with_bias = self.add_bias_term(X_test)
        else:
            X_test_with_bias = None

        for epoch in range(self.epochs):
            y_pred = self.sigmoid(np.dot(X_with_bias, self.weights))
            error = y_pred - y.reshape(-1, 1)
            gradient = np.dot(X_with_bias.T, error) / m
            self.weights -= self.learning_rate * gradient

            # Calculate training loss and error
            train_predictions = self.sigmoid(np.dot(X_with_bias, self.weights))
            train_loss = self.compute_loss(y, train_predictions)
            train_error = np.mean((train_predictions >= 0.5) != y.reshape(-1, 1))
            self.train_losses.append(train_loss)
            self.train_errors.append(train_error)

            # Calculate test loss and error if the test set is provided
            if X_test_with_bias is not None and y_test is not None:
                test_predictions = self.predict(X_test_with_bias)
                test_loss = self.compute_loss(y_test, test_predictions)
                test_error = np.mean((test_predictions >= 0.5) != y_test.reshape(-1, 1))
                self.test_losses.append(test_loss)
                self.test_errors.append(test_error)

            # Check for convergence
            if np.linalg.norm(self.learning_rate * gradient) < self.convergence_threshold:
                break

    def train(self, X, y, X_test=None, y_test=None):
        X_with_bias = self.add_bias_term(X)
        if X_test is not None:
            X_test_with_bias = self.add_bias_term(X_test)
        else:
            X_test_with_bias = None

        self.gradient_descent(X_with_bias, y, X_test=X_test_with_bias, y_test=y_test)

    def predict(self, X):
        X_with_bias = self.add_bias_term(X)
        predictions = self.sigmoid(np.dot(X_with_bias, self.weights))
        return (predictions >= 0.5).astype(int)

    def plot_errors_and_losses(self):
        epochs_range = range(1, len(self.train_errors) + 1)

        # Plot test error vs epochs
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range[-5:], self.test_errors[-5:], marker='o')
        plt.title('Test Error vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Error')

        # Plot training error vs epochs
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self.train_errors, marker='o')
        plt.title('Training Error vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Error')

        plt.tight_layout()
        plt.show()

        # Plot test loss vs epochs
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range[-5:], self.test_losses[-5:], marker='o')
        plt.title('Test Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Plot training loss vs epochs
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self.train_losses, marker='o')
        plt.title('Training Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.show()

# Load MNIST dataset using torchvision
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Extract data and labels
X_train = train_dataset.data.numpy().reshape(-1, 28 * 28)
y_train = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy().reshape(-1, 28 * 28)
y_test = test_dataset.targets.numpy()

# Keep only the first two classes
X_train = X_train[(y_train == 0) | (y_train == 1)]
y_train = y_train[(y_train == 0) | (y_train == 1)]
X_test = X_test[(y_test == 0) | (y_test == 1)]
y_test = y_test[(y_test == 0) | (y_test == 1)]

# Apply PCA to reduce dimensionality to 2
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Example usage:
logistic_reg = LogisticRegression(learning_rate=0.1, epochs=100, convergence_threshold=1e-5)
logistic_reg.train(X_train_pca, y_train, X_test_pca, y_test)

logistic_reg.plot_errors_and_losses()
