"""
SYDE 572 - Assignment 4

Author: Karla Castro

Question 3- extra points

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

# Define the MLP (three layer) neural network

# Set random seed for reproducibility
np.random.seed(28)
torch.manual_seed(28)

class MLP(nn.Module):
    def __init__(self, size1=784, size2=512, size3=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() # flatten the input
        # first layer
        self.hidden1 = nn.Linear(size1, size2)
        self.act1 = nn.ReLU() # ReLU activation function
        # second layer
        self.hidden2 = nn.Linear(size2, size2)
        self.bn1 = nn.BatchNorm1d(size2) # Batch normalization
        self.act2 = nn.ReLU() # ReLU activation function
        # third layer
        self.hidden3 = nn.Linear(size2, size2)
        self.bn2 = nn.BatchNorm1d(size2) # Batch normalization
        self.act3 = nn.ReLU() # ReLU activation function
        # fourth layer
        self.hidden4 = nn.Linear(size2, size3)

     # Forward function defining how nework will process when we paass an input trhough it 
    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.bn1(x)
        x = self.act2(x)
        x = self.hidden3(x)
        x = self.bn2(x)
        x = self.act3(x)
        x = self.hidden4(x)
        return x
    
# Define the CNN neural network
MLP_model = MLP() # create an instance of the model
print(MLP_model) 

# Train it
loss_function = nn.CrossEntropyLoss() # Cross Entropy Loss
optimizer = optim.SGD(MLP_model.parameters(), lr = 0.01, momentum=0.9) # SGD optimizer

# Training loop
epochs = 20
batch_size = 64

# Define dataset

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Set up data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []

for epoch in range(epochs):
    # accumulated loss for each epoch
    # cont correct predictions for each epoch
    # count total train samples in each epoch
    MLP_model.train()
    accumulated_loss, correct_pred, total_train = 0.0, 0, 0

    for x_batch, y_batch in train_loader:
        y_predict = MLP_model(x_batch)
        loss = loss_function(y_predict, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accumulated_loss += loss.item() # loss.item() returns the value held in the loss
        _, y_pred = torch.max(y_predict.data, 1) # torch.max returns the maximum value of all elements in the input tensor
        correct_pred += (y_pred == y_batch).sum().item()
        total_train += y_batch.size(0)

    train_losses.append(accumulated_loss/len(train_loader))
    train_accuracies.append(correct_pred/total_train)
    
    # Test loop
    MLP_model.eval()
    accumulated_loss, correct_pred, total_test = 0.0, 0, 0

    for x_batch, y_batch in test_loader:
        y_predict = MLP_model(x_batch)
        loss = loss_function(y_predict, y_batch)

        accumulated_loss += loss.item()
        _, y_pred = torch.max(y_predict.data, 1)
        correct_pred += (y_pred == y_batch).sum().item()
        total_test += y_batch.size(0)

    test_losses.append(accumulated_loss)
    test_accuracies.append(correct_pred/total_test)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}')

    def plot_MLP_model(train_losses, train_accuracies, test_losses, test_accuracies):
        # Plot loss
        epochs = range(len(train_losses)-5, len(train_losses))# last 5 epochs
        num_last_epochs = len(epochs)

        plt.plot(epochs, train_losses[-num_last_epochs:], label='Training loss')
        plt.title('Loss vs Epochs Training Dataset')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.plot(epochs, test_losses[-num_last_epochs:], label='Test loss', color='orange')
        plt.title('Loss vs Epochs Test Dataset')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Plot accuracy
        plt.plot(epochs, train_accuracies[-num_last_epochs:], label='Training accuracy')
        plt.title('Accuracy vs Epochs Training Dataset')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.plot(epochs, test_accuracies[-num_last_epochs:], label='Test accuracy', color='orange')
        plt.title('Accuracy vs Epochs Test Dataset')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

plot_MLP_model(train_losses, train_accuracies, test_losses, test_accuracies)