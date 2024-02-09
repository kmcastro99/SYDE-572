"""
SYDE 572 - Assignment 3

Author: Karla Castro

Question 1

"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

def histogram_based_estimations(data, region_sizes):
    """
    Function to find the histogram-based estimations for each class
    Parameters:
        - data: dataset to find the histogram-based estimations
        - class_name: class name of the dataset
        - region_sizes: histogram region sizes
        - color_hist: color of the histogram
    Return: 
        - histogram-based estimations
    """
    # Histogram-based estimation points
    # Create an empty matrix for finding the number of bins given a region size
    bins_number = int((np.max(data)-np.min(data))/region_sizes)+1
    # Create a histogram with the probability values
    # Set density to true to normalize the histogram
    hist, bins = np.histogram(data, bins=bins_number, density=True)
    return hist, bins
    
def plot_histogram(data, num_bins, class_name, region_sizes, color_hist = None):
    """
    Function to plot the histogram distribution
    Parameters:
        - data: dataset to plot the histogram distribution
        - num_bins: number of bins
        - class_name: class name of the dataset
        - region_sizes: histogram region sizes
        - color_hist: color of the histogram
        Return:
        - histogram distribution
        """
    # Plot histogram distribution
    # set density to true to normalize the histogram
    plt.hist(data, bins=num_bins, color=color_hist, alpha=0.7, density=True)
    plt.title(f'Probability Distribution of {class_name} with Region Size: {region_sizes}')
    plt.xlabel('Data Values (x)')
    plt.ylabel('Probability (P(x))')
    plt.show()

# Maximum Likelihood (ML) Classifier using histogram-based estimation
def ML_classifier(hist_c0, hist_c1, bins_c0, bins_c1, x):
    """
    Function to find the ML classifier using histogram-based probability distribution
    Parameters:
        - x: data from which the histogram was created
        - hist: probabilities of the histogram
        - bins: bins edges used for the histogram
    Return: 
        - 0 if class 0
        - 1 if class 1
    """
     # Calculate the probability of the sample belonging to class 1
     # Interpolate the likelihood for the new_data_point based on the provided histogram values and bin edges
    likelihood_c0 = np.interp(x, bins_c0[:-1], hist_c0)
    # Calculate the probability of the sample belonging to class 1
    likelihood_c1 = np.interp(x, bins_c1[:-1], hist_c1)
    # return likelihood
    if likelihood_c0 > likelihood_c1:
        predicted_class = 0 # Class 0
    else:
        predicted_class = 1# Class 1

    return predicted_class

# Function to calculate accuracy of model
def calculate_accuracy (class_label, predicted_values, num_samples):
    """
    Function to calculate accuracy of model
    Parameters:
        - class_label: class label of the dataset
        - predicted_values: predicted values of the dataset
        - num_samples: number of samples in the dataset
        Return:
        - classification accuracy"""
    # Counter for correct predictions
    count_correct_predictions = 0

    # Check if the predicted values belong to the class_label
    for point in predicted_values:
        if point == class_label:
            # Add to counter if a label belongs to the class
            count_correct_predictions += 1

    # Report the classification accuracy
    classification_accuracy = (count_correct_predictions/num_samples)*100
    return classification_accuracy

# Function for kernel-based density estimation
# Giiven a gaussian kernel

def kernel_based_density_estimation(data, x, h=20):
    """
    Function for kernel-based density estimation
    Parameters:
        - data: data points for which to estimate the density
        - x: points at which to estimate the density
        - h: scaling factor of the Gaussian kernel
    Return: 
        - kernel-based density estimation at x values
    """
    density_estimation = np.zeros(len(x))

    for i, x in enumerate(x):
        # Calculate the Gaussian kernel values for each data point
        kernel_values = np.exp(-0.5 * ((data - x) / h) ** 2) / (h * np.sqrt(2 * np.pi))
        # Estimate the density at the current x value
        density_estimation[i] = np.sum(kernel_values) / (len(data) * h)
        # density_estimation = density_estimation / np.sum(density_estimation)
    return density_estimation

def ML_classifier_kernel_estimation(class_0, class_1, x):
    """
    Function to find the ML classifier using kernel-based density estimation
    Parameters:
        - density_class_0: density estimation of class 0
        - density_class_1: density estimation of class 1
    Return: 
        - 0 if class 0
        - 1 if class 1
    """
    density_class_0 = kernel_based_density_estimation(class_0, x)
    density_class_1 = kernel_based_density_estimation(class_1, x)

    if density_class_0 > density_class_1:

        predicted_class = 0 # Class 0
    else:
        predicted_class = 1 # Class 1

    return predicted_class

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# Define a data transformation to preprocess the images and flatten them
transform = transforms.Compose([transforms.ToTensor()])

# Create two datasets: one for training and one for testing using the torchvision.datasets.MNIST class
# Training dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# Test dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Classes to keep when filtering the training dataset 
classes_to_keep = [0, 1]

# Flatten the images (train_loader and test_loader) from 28x28 to 784x1 vectors
train_images = train_dataset.data.reshape(-1, 784)
test_images = test_dataset.data.reshape(-1, 784)

# Convert the 784x1 vectors to 2x1 vectors using PCA
pca = PCA(n_components=1)
train_images_pca = pca.fit_transform(train_images)
test_images_pca = pca.fit_transform(test_images)
# print(train_images_pca.shape) # returns a column vector of 60000x1

# Estimate the probability of the dataset using histogram-based estimation
# Separate the dataset into two classes: 0 and 1
classes_labels = ['Class 0', 'Class 1']
class_0 = train_images_pca[train_dataset.targets == classes_to_keep[0]] # Class with values 0
class_1 = train_images_pca[train_dataset.targets == classes_to_keep[1]] # Class with values 1
# Test dataset
class_0_test = test_images_pca[test_dataset.targets == classes_to_keep[0]] # Class with values 0
class_1_test = test_images_pca[test_dataset.targets == classes_to_keep[1]] # Class with values 1

# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# Histogram based Estimation

# Histogram region sizes
region_size = [1, 10, 100]

# Call function to find the histogram-based estimations for each class
# Histogram-based estimation for class 0, region 1
hist_c0_r1 = histogram_based_estimations(class_0, region_size[0])
# Histogram-based estimation for class 0, region 10
hist_c0_r10 = histogram_based_estimations(class_0, region_size[1])
# Histogram-based estimation for class 0, region 100
hist_c0_r100 = histogram_based_estimations(class_0, region_size[2])

# Histogram-based estimation for class 1, region 1
hist_c1_r1= histogram_based_estimations(class_1, region_size[0])
# Histogram-based estimation for class 1, region 10
hist_c1_r10 = histogram_based_estimations(class_1, region_size[1])
# Histogram-based estimation for class 1, region 100
hist_c1_r100 = histogram_based_estimations(class_1, region_size[2])

# # Plot histogram distribution
# Histogram-based estimation for class 0
plot_histogram(class_0, hist_c0_r1[1], classes_labels[0], region_size[0], color_hist = 'blue')
plot_histogram(class_0, hist_c0_r10[1], classes_labels[0], region_size[1], color_hist = 'blue')
plot_histogram(class_0, hist_c0_r100[1], classes_labels[0], region_size[2], color_hist = 'blue')

# Histogram-based estimation for class 1
plot_histogram(class_1, hist_c1_r1[1], classes_labels[1], region_size[0], color_hist = 'red')
plot_histogram(class_1, hist_c1_r10[1], classes_labels[1], region_size[1], color_hist = 'red')
plot_histogram(class_1, hist_c1_r100[1], classes_labels[1], region_size[2], color_hist = 'red')

# Find the ML classifier for each region size
# ML classifier for region size 1
# Class 0
prob_c0_r1 = [ML_classifier(hist_c0_r1[0], hist_c1_r1[0], hist_c0_r1[1], hist_c1_r1[1], x) for x in class_0_test]
accuracy = calculate_accuracy(0, prob_c0_r1, len(class_0_test))
print(f'Accuracy for class 0 with region size 1: {accuracy}')
# Class 1
prob_c1_r1 = [ML_classifier(hist_c0_r1[0], hist_c1_r1[0], hist_c0_r1[1], hist_c1_r1[1], x) for x in class_1_test]
accuracy = calculate_accuracy(1, prob_c1_r1, len(class_1_test))
print(f'Accuracy for class 1 with region size 1: {accuracy}')
# ML classifier for region size 10
# Class 0
prob_c0_r10 = [ML_classifier(hist_c0_r10[0], hist_c1_r10[0], hist_c0_r10[1], hist_c1_r10[1], x) for x in class_0_test]
accuracy = calculate_accuracy(0, prob_c0_r10, len(class_0_test))
print(f'Accuracy for class 0 with region size 10: {accuracy}')
# Class 1
prob_c1_r10 = [ML_classifier(hist_c0_r10[0], hist_c1_r10[0], hist_c0_r10[1], hist_c1_r10[1], x) for x in class_1_test]
accuracy = calculate_accuracy(1, prob_c1_r10, len(class_1_test))
print(f'Accuracy for class 1 with region size 10: {accuracy}')
# ML classifier for region size 100
# Class 0
prob_c0_r100 = [ML_classifier(hist_c0_r100[0], hist_c1_r100[0], hist_c0_r100[1], hist_c1_r100[1], x) for x in class_0_test]
accuracy = calculate_accuracy(0, prob_c0_r100, len(class_0_test))
print(f'Accuracy for class 0 with region size 100: {accuracy}')
# Class 1
prob_c1_r100 = [ML_classifier(hist_c0_r100[0], hist_c1_r100[0], hist_c0_r100[1], hist_c1_r100[1], x) for x in class_1_test]
accuracy = calculate_accuracy(1, prob_c1_r100, len(class_1_test))
print(f'Accuracy for class 1 with region size 100: {accuracy}')

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# Kernel Estimation
# Estimating the density

values_x_c0 = np.linspace(min(class_0_test), max(class_0_test), len(class_0_test))
values_x_c1 = np.linspace(min(class_1_test), max(class_1_test), len(class_1_test))
density_class_0 = [kernel_based_density_estimation(class_0, x) for x in values_x_c0]
density_class_0_n = density_class_0 / np.sum(density_class_0)
density_class_1 = [kernel_based_density_estimation(class_1, x) for x in values_x_c1]
density_class_1_n = density_class_1 / np.sum(density_class_1)

# Find the ML classifier using kernel estimation for both test classes 
prob_c0 = [ML_classifier_kernel_estimation(class_0, class_1, x) for x in class_0_test]
prob_c1 = [ML_classifier_kernel_estimation(class_0, class_1, x) for x in class_1_test]

# Find the accuracy for each class
accuracy_c0 = calculate_accuracy(0, prob_c0, len(class_0_test))
accuracy_c1 = calculate_accuracy(1, prob_c1, len(class_1_test))
print(f'Accuracy for class 0 with kernel-based density estimation: {accuracy_c0}')
print(f'Accuracy for class 1 with kernel-based density estimation: {accuracy_c1}')

# Plot the density
plt.plot(values_x_c0, density_class_0, label='Kernel Density Estimation Class 0', color='blue')
plt.xlabel('X Values')
plt.ylabel('Density')
plt.title('Kernel Density Estimation')
plt.legend()
plt.show()


plt.plot(values_x_c1, density_class_1, label='Kernel Density Estimation Class 1', color='red')
plt.xlabel('X Values')
plt.ylabel('Density')
plt.title('Kernel Density Estimation')
plt.legend()
plt.show()
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# Using Parametric Estimation Approach

# Create an ML (maximum likelihood) classifier that combines a Gaussian and an exponential distribution
def estimate_pdf(x, mean, cov, inv_cov):
    # Calculate the PDF for multivariate normal distribution
    exponent = -0.5 * np.dot(np.dot((x-mean), inv_cov), (x-mean))
    probability = np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    return probability

# Maximum Likelihood (ML) Classifier (0 for C0 and 1 for C1)
def ML_classifier_parametric(x, mean_c1, mean_c2, cov_c1, cov_c2, inv_cov_c1, inv_cov_c2):
    # Calculate the likelihood of the sample belonging to class 1
    likelihood_c1 = estimate_pdf(x, mean_c1, cov_c1, inv_cov_c1)
    # Calculate the likelihood of the sample belonging to class 2
    likelihood_c2 = estimate_pdf(x, mean_c2, cov_c2, inv_cov_c2)
    # Compare the likelihoods and assign the label to the sample
    if likelihood_c1 > likelihood_c2:
        return 0 # Class 0
    else:
        return 1 # Class 1

# Find the ML classifier using parametric estimation for both test classes
# Calculate the mean for each class
mean_class_0 = np.mean(class_0)
mean_class_1 = np.mean(class_1)

# Calculate the covariance matrix for each class
cov_class_0 = np.array([[np.cov(class_0, rowvar = False)]])
cov_class_1 = np.array([[np.cov(class_1, rowvar = False)]])

# Calculate the inverse of the covariance matrix for each class
inv_cov_class_0 = 1.0 / cov_class_0
inv_cov_class_1 = 1.0 / cov_class_1

# Class 0
likelihood_c0 = [ML_classifier_parametric(x, mean_class_0, mean_class_1, cov_class_0, cov_class_1, inv_cov_class_0, inv_cov_class_1) for x in class_0_test]
# # Class 1
likelihood_c1 = [ML_classifier_parametric(x, mean_class_0, mean_class_1, cov_class_0, cov_class_1, inv_cov_class_0, inv_cov_class_1) for x in class_1_test]

# Find the accuracy for each class
accuracy_c0 = calculate_accuracy(0, likelihood_c0, len(class_0_test))
accuracy_c1 = calculate_accuracy(1, likelihood_c1, len(class_1_test))
print(f'Accuracy for class 0 with parametric estimation: {accuracy_c0}')
print(f'Accuracy for class 1 with parametric estimation: {accuracy_c1}')