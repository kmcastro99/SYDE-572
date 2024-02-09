import numpy as np
import matplotlib.pyplot as plt

"""
SYDE 572 - Assignment 2

Author: Karla Castro

Question 3

"""
np.random.seed(0)

# Create an ML (maximum likelihood) classifier with an exponential distribution
def ML_classifier_exponential_distribution(x, mean_c1, mean_c2):

    # Calculate lambda for each class
    lambda_c1 = 1.0 / mean_c1
    lambda_c2 = 1.0 / mean_c2

    likelihood_c1 = lambda_c1 * np.exp(-lambda_c1 * x)
    likelihood_c2 = lambda_c2 * np.exp(-lambda_c2 * x)

    if np.all(likelihood_c1 > likelihood_c2):
        return 0 # Class 1
    else:
        return 1 # Class 2

# Create an ML (maximum likelihood) classifier with an uniform distribution
def ML_classifier_uniform_distribution(x):

    # Define a and b
    # Where a is the min value of x
    a = np.min(x)
    # Where b is the max value of x
    b = np.max(x)

    condition1 = np.logical_and(a <= x, x <= b)
    condition2 = np.logical_and(b <= x, x <= a)
    
    class1_condition = condition1.any()
    class2_condition = condition2.any()

    if class1_condition:
        return 0  # Class 1
    elif class2_condition:
        return 1  # Class 2

# Create an ML (maximum likelihood) classifier that combines a Gaussian and an exponential distribution
def ML_classifier_gaussian_exponential_distribution(x, mean_c1, mean_c2, sigma_c1, sigma_c2):

    # Calculate lambda for each class
    lambda_c1 = 1.0 / mean_c1
    lambda_c2 = 1.0 / mean_c1

    # Calculate exponential distribution for each class
    expo_c1 = lambda_c1 * np.exp(-lambda_c1 * x)
    expo_c2 = lambda_c2 * np.exp(-lambda_c2 * x)

    # Calculate the Gaussian distribution for each class
    gaussian_c1 = (1.0 / np.sqrt(2 * np.pi * sigma_c1)) * np.exp(-((x - mean_c1)**2 / (2 * sigma_c1)))
    gaussian_c2 = (1.0 / np.sqrt(2 * np.pi * sigma_c2)) * np.exp(-((x - mean_c2)**2 / (2 * sigma_c2)))

    # Calculate the probability for each class
    probability_c1 = (1/2) * expo_c1 * gaussian_c1 # Given formula in homework
    probability_c2 = (1/2) * expo_c2 * gaussian_c2

    if np.all(probability_c1 > probability_c2):
        return 0 # Class 1
    else:
        return 1 # Class 2

# Function to calculate accuracy of model
def calculate_accuracy (predicted_values, num_samples):

    # Counter for correct predictions
    count_correct_predictions = 0

    # Check if the first 50 labels belong to C1
    for point in predicted_values[:50]:
        if point == 0:
            # Add to counter if a label belongs to C1
            count_correct_predictions += 1
    # Check if the last 50 labels belong to C2
    for point in predicted_values[-50:]:
        # Add to counter if a label belongs to C2
        if point == 1:
            count_correct_predictions += 1

    # Report the classification accuracy
    total_data_test = num_samples
    error_classifier = count_correct_predictions/total_data_test
    return error_classifier

# Generate 100 samples from a Gaussian distribution

num_samples = 100
# Class 1
mean_c1 = 0.5
sigma_c1 = 1

# Class 2
mean_c2 = 5
sigma_c2 = 3

# Generate random samples from a normal (Gaussian) distribution
samples_c1 = np.random.normal(loc = mean_c1, scale = sigma_c1, size = num_samples)
samples_c2 = np.random.normal(loc = mean_c2, scale = sigma_c2, size = num_samples)

# Calculate mean for random samples
mean_samples_c1 = np.mean(samples_c1)
mean_samples_c2 = np.mean(samples_c2)

# Calculate covariance for random samples
sigma_samples_c1 = np.cov(samples_c1)
sigma_samples_c2 = np.cov(samples_c2)

# Generate 50 samples for each class with white noise
# Class 1
mean_noise = 0
sigma_noise = 1

noise_c1 = np.random.normal(loc = mean_noise, scale = sigma_noise, size = 50)
noise_c2 = np.random.normal(loc = mean_noise, scale = sigma_noise, size = 50)
noise = np.hstack((noise_c1, noise_c2))

# Generate random samples from a normal (Gaussian) distribution
test_samples_c1 = np.random.normal(loc = mean_c1, scale = sigma_c1, size = 50)
test_samples_c2 = np.random.normal(loc = mean_c2, scale = sigma_c2, size = 50)
samples = np.hstack((test_samples_c1, test_samples_c2))
test_samples = samples + noise

# Test the ML classifier with an exponential distribution
predictions_exponential = [ML_classifier_exponential_distribution(point, mean_samples_c1, mean_samples_c2) for point in test_samples]

# Test the ML classifier with an uniform distribution
predictions_uniform = [ML_classifier_uniform_distribution(point) for point in test_samples]

# Test the ML classifier that combines a Gaussian and an exponential distribution
predictions_gaussian_exponential = [ML_classifier_gaussian_exponential_distribution(point, mean_samples_c1, mean_samples_c2, sigma_samples_c1, sigma_samples_c2) for point in test_samples]

# Calculate accuracy for each model
accuracy_exponential = calculate_accuracy(predictions_exponential, 100)
accuracy_uniform = calculate_accuracy(predictions_uniform, 100)
accuracy_gaussian_exponential = calculate_accuracy(predictions_gaussian_exponential, 100)

# Find the accuracy for each model
print("Accuracy for ML classifier with an exponential distribution: ", accuracy_exponential*100 , '%')
print("Accuracy for ML classifier with an uniform distribution: ", accuracy_uniform*100, '%')
print("Accuracy for ML classifier that combines a Gaussian and an exponential distribution: ", accuracy_gaussian_exponential*100, '%')








