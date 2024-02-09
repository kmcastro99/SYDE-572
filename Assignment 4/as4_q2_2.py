# -*- coding: utf-8 -*-
"""
SYDE 572 - Assignment 4

Author: Karla Castro

Question 2 - part 2

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
np.random.seed(28)

# Check for GPU availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define VGG11 architecture for MNIST
class VGG11_adam(nn.Module):

  def __init__(self):
    super(VGG11_adam, self).__init__()
    self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    self.bn1_1 = nn.BatchNorm2d(64)
    self.relu1_1 = nn.ReLU()
    self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn2_1 = nn.BatchNorm2d(128)
    self.relu2_1 = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.bn3_1 = nn.BatchNorm2d(256)
    self.relu3_1 = nn.ReLU()

    self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.bn3_2 = nn.BatchNorm2d(256)
    self.relu3_2 = nn.ReLU()
    self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.bn4_1 = nn.BatchNorm2d(512)
    self.relu4_1 = nn.ReLU()

    self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn4_2 = nn.BatchNorm2d(512)
    self.relu4_2 = nn.ReLU()
    self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn5_1 = nn.BatchNorm2d(512)
    self.relu5_1 = nn.ReLU()

    self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn5_2 = nn.BatchNorm2d(512)
    self.relu5_2 = nn.ReLU()
    self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    # fully connected layer 1
    self.fc1 = nn.Linear(512, 4096)
    self.relu_fc1 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)

    # fully connected layer 2
    self.fc2 = nn.Linear(4096, 4096)
    self.relu_fc2 = nn.ReLU()
    self.dropout2 = nn.Dropout(0.5)

    self.fc3 = nn.Linear(4096, 10)

  def forward(self,x):
    x = self.conv1_1(x)
    x = self.bn1_1(x)
    x = self.relu1_1(x)
    x = self.maxpool1(x)

    x = self.conv2_1(x)
    x = self.bn2_1(x)
    x = self.relu2_1(x)
    x = self.maxpool2(x)

    x = self.conv3_1(x)
    x = self.bn3_1(x)
    x = self.relu3_1(x)
    x = self.maxpool3(x)

    x = self.conv4_1(x)
    x = self.bn4_1(x)
    x = self.relu4_1(x)
    x = self.conv4_2(x)
    x = self.bn4_2(x)
    x = self.relu4_2(x)
    x = self.maxpool4(x)

    x = self.conv5_1(x)
    x = self.bn5_1(x)
    x = self.relu5_1(x)
    x = self.conv5_2(x)
    x = self.bn5_2(x)
    x = self.relu5_2(x)
    x = self.maxpool5(x)

    x = x.view(x.size(0), -1)  # Flatten
    x = self.relu_fc1(self.fc1(x))
    x = self.dropout1(x)

    x = self.relu_fc2(self.fc2(x))
    x = self.dropout2(x)

    x = self.fc3(x)

    return x

# Load the MNIST dataset
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Set up data loaders
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model, loss function, and optimizer
model_adam = VGG11_adam().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_adam.parameters(), lr=0.001)
print(model_adam)

# Train
epochs = 20
train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []

for epoch in range(epochs):
    # accumulated loss for each epoch
    # cont correct predictions for each epoch
    # count total train samples in each epoch
    model_adam.train()
    accumulated_loss, correct_pred, total_train = 0.0, 0, 0

    for x_batch, y_batch in train_loader:
        # Move Data to GPU
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_predict = model_adam(x_batch)
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
    model_adam.eval()
    accumulated_loss, correct_pred, total_test = 0.0, 0, 0

    for x_batch, y_batch in test_loader:
        # Move Data to GPU
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_predict = model_adam(x_batch)
        loss = loss_function(y_predict, y_batch)

        accumulated_loss += loss.item()
        _, y_pred = torch.max(y_predict.data, 1)
        correct_pred += (y_pred == y_batch).sum().item()
        total_test += y_batch.size(0)

    test_losses.append(accumulated_loss)
    test_accuracies.append(correct_pred/total_test)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]*100:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]*100:.4f}')

# Training accuracy vs the number of epochs
epochs=20
epochs = np.arange(epochs)
plt.plot(epochs[-5:], train_accuracies[-5:], label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs Epochs with Adam Optimizer')
plt.legend()
plt.show()

plt.show()#Test accuracy vs the number of epochs
plt.plot(epochs[-5:], test_accuracies[-5:], label='Test Accuracy',color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy vs Epochs with Adam Optimizer')
plt.legend()
plt.show()

# Training loss vs the number of epochs
plt.plot(epochs[-5:], train_losses[-5:], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss vs Epochs with Adam Optimizer')
plt.legend()
plt.show()

# Test loss vs the number of epochs
plt.plot(epochs[-5:], test_losses[-5:], label='Test Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss vs Epochs with Adam Optimizer')
plt.legend()
plt.show()