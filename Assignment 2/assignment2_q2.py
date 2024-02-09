import numpy as np
import matplotlib.pyplot as plt

"""
SYDE 572 - Assignment 2

Author: Karla Castro

Question 2

"""
np.random.seed(0)

# Create function to generate random samples
def generate_random_samples(num_samples, mean, covariance):
    """
    Generates random samples given the mean and covariance
    """
    # Create empty matrix where the generated samples will be saved
    generated_samples = np.round(np.random.multivariate_normal(mean, covariance, size = num_samples),2)
    return generated_samples

# Create function to add noise to samples
def add_noise(samples1, samples2, mean, cov, num_samples):

    # Generate 10 noisy samples 
    noisy_samples = np.round(np.random.multivariate_normal(mean, cov, size = num_samples*2),2)

    # Create empty matrix to append new samples (addition result) for each class 
    new_samples_c1 = [] # Class 1
    new_samples_c2 = [] # Class 2

    # Iterate over the generated samples of class 1 to add the first 5 noisy samples
    for i in range(len(samples1)):
        noisy_sample = noisy_samples[:num_samples]
        new_sample_c1 = np.round(samples1[i]+noisy_sample[i],2)
        new_samples_c1.append(new_sample_c1)
    # Iterate over the generated samples of class 1 to add the last 5 noisy samples
    for i in range(len(samples2)):
        noisy_sample = noisy_samples[-num_samples:]
        new_sample_c2 = np.round(samples2[i]+noisy_sample[i],2)
        new_samples_c2.append(new_sample_c2)
    
    return new_samples_c1, new_samples_c2

# Estimate the probability distributions of the two classes (both have random normal distribution)
def estimate_pdf(x, mean, cov, inv_cov):
    # Calculate the PDF for multivariate normal distribution
    likelihood = (1.0/(2*np.pi*np.sqrt(np.linalg.det(cov))))*np.exp(-0.5*np.dot(np.dot((x-mean),inv_cov),(x-mean).T))
    return likelihood

# Maximum Likelihood (ML) Classifier (0 for C1 and 1 for C2)
def ML_classifier(x, mean_c1, mean_c2, cov_c1, cov_c2, inv_cov_c1, inv_cov_c2):
    # Calculate the likelihood of the sample belonging to class 1
    likelihood_c1 = np.round(estimate_pdf(x, mean_c1, cov_c1, inv_cov_c1),2)
    # Calculate the likelihood of the sample belonging to class 2
    likelihood_c2 = np.round(estimate_pdf(x, mean_c2, cov_c2, inv_cov_c2),2)
    # Compare the likelihoods and assign the label to the sample
    if likelihood_c1 > likelihood_c2:
        return 0 # Class 1
    else:
        return 1 # Class 2

# Maximum A posteriori (MAP) Classifier (0 for C1 and 1 for C2)
def MAP_classifier(x, mean_c1, mean_c2, cov_c1, cov_c2, inv_cov_c1, inv_cov_c2, pc1, pc2):
    # Find the likelihood of the sample belonging to class 1
    likelihood_c1 = estimate_pdf(x, mean_c1, cov_c1, inv_cov_c1)
    # Find the likelihood of the sample belonging to class 2
    likelihood_c2 = estimate_pdf(x, mean_c2, cov_c2, inv_cov_c2)
    # Find the posterior probability of the sample belonging to class 1
    posterior_c1 = np.round((likelihood_c1*pc1)/(likelihood_c1*pc1 + likelihood_c2*pc2),2)
    # Find the posterior probability of the sample belonging to class 2
    posterior_c2 = np.round((likelihood_c2*pc2)/(likelihood_c1*pc1 + likelihood_c2*pc2),2)
    # Compare the posterior probabilities and assign the label to the sample
    if posterior_c1 > posterior_c2:
        return 0 # Class 1
    else:
        return 1 # Class 2

# Function to calculate accuracy of model
def calculate_accuracy (predicted_values, num_samples):

    # Counter for correct predictions
    count_correct_predictions = 0

    # Check if the first 50 labels belong to C1
    for point in predicted_values[:num_samples]:
        if point == 0:
            # Add to counter if a label belongs to C1
            count_correct_predictions += 1
    # Check if the last 50 labels belong to C2
    for point in predicted_values[-num_samples:]:
        # Add to counter if a label belongs to C2
        if point == 1:
            count_correct_predictions += 1

    # Report the classification accuracy
    total_data_test = num_samples*2
    error_classifier = count_correct_predictions/total_data_test
    return error_classifier

# Define the mean vector (for 100 samples)
# Class 1
mean_vector_1 = np.array([1, 3])
# Class 2
mean_vector_2 = np.array([4, 7])

# Define the summation matrix
# Class 1
covariance_matrix_1 = np.array([[1,0],[0,15]])
# Class 2
covariance_matrix_2 = np.array([[3,4],[4,11]])

# New mean vector to generate noisy samples
mean_vector_new = np.array([2, 2])
# New covariance matrix to generate noisy samples
covariance_matrix_new = np.array([[2, 0], [0, 3]])

# Define number of samples
num_samples_50 = 50 # For question asking for 50 samples with noise
num_samples_100 = 100 # For question asking for 100 samples

# Generate samples for the classes above
generated_samples_c1_100 = generate_random_samples(num_samples_100, mean_vector_1, covariance_matrix_1) # Class 1 samples (100 samples)
generated_samples_c2_100 = generate_random_samples(num_samples_100, mean_vector_2, covariance_matrix_2) # Class 2 samples (100 samples)

# Generate noisy samples to add to each sample from generated_samples_c1 and generated_samples_c2
new_samples_c1_100, new_samples_c2_100 = add_noise(generated_samples_c1_100, generated_samples_c2_100, mean_vector_new, covariance_matrix_new, num_samples_100)

# Calculate the sample mean vector
mean_vector_c1_100 = np.round(np.mean(new_samples_c1_100, axis=0),2) # Class 1
mean_vector_c2_100 = np.round(np.mean(new_samples_c2_100, axis=0),2) # Class 2

# Calculate the sample covariance matrix
covariance_matrix_c1_100 = np.round(np.cov(new_samples_c1_100, rowvar = False),2) # Class 1
covariance_matrix_c2_100 = np.round(np.cov(new_samples_c2_100, rowvar = False),2) # Class 2

inv_cov_c1_100 = np.round(np.linalg.inv(covariance_matrix_c1_100),2) # Class 1
inv_cov_c2_100= np.round(np.linalg.inv(covariance_matrix_c2_100),2) # Class 2

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Estimate the probability distributions of the two classes (both have random normal distribution)
pdf_c1 = estimate_pdf(new_samples_c1_100, mean_vector_c1_100, covariance_matrix_c1_100, inv_cov_c1_100) # Class 1
pdf_c2 = estimate_pdf(new_samples_c2_100, mean_vector_c2_100, covariance_matrix_c2_100, inv_cov_c2_100) # Class 2

print( 'The probabily distribution of class 1 is: ', pdf_c1)
print( 'The probabily distribution of class 2 is: ', pdf_c2)

# Generate 50 samples with white noise that will be tested using the ML Classifier
# Generate samples for the classes above
generated_samples_c1_50 = generate_random_samples(num_samples_50, mean_vector_1, covariance_matrix_1) # Class 1 samples (50 samples)
generated_samples_c2_50 = generate_random_samples(num_samples_50, mean_vector_2, covariance_matrix_2) # Class 2 samples (50 samples)

# Generate noisy samples to add to each sample from generated_samples_c1 and generated_samples_c2
new_samples_c1_50, new_samples_c2_50 = add_noise(generated_samples_c1_50, generated_samples_c2_50, mean_vector_new, covariance_matrix_new, num_samples_50)

# Add white noise 
mean_white_noise = np.array([0, 0])
cov_white_noise = np.array([[1, 0], [0, 1]])

# Call function to add noise to samples
white_noise_c1_50, white_noise_c2_50 = add_noise(new_samples_c1_50, new_samples_c2_50, mean_white_noise, cov_white_noise, num_samples_50)
# Stack vertically noise samples of class 1 and class 2
test_samples = np.vstack((white_noise_c1_50, white_noise_c2_50))

# Call ML classifier function to classify noisy samples using the training data of 100 samples
predictions_ML = np.array([ML_classifier(x, mean_vector_c1_100, mean_vector_c2_100, covariance_matrix_c1_100, covariance_matrix_c2_100, inv_cov_c1_100, inv_cov_c2_100) for x in test_samples])
# Calculate the accuracy of the model
accuracy_ML = calculate_accuracy(predictions_ML, num_samples_50)
print('The accuracy of the ML classifier is: ', accuracy_ML*100, '%')

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Given two probabilities for each class
p_c1 = 0.58
p_c2 = 0.42

# Call MAP classifier function to classify noisy samples using the training data of 100 samples
predictions_MAP = np.array([MAP_classifier(x, mean_vector_c1_100, mean_vector_c2_100, covariance_matrix_c1_100, covariance_matrix_c2_100, inv_cov_c1_100, inv_cov_c2_100, p_c1, p_c2) for x in test_samples])

# Calculate the accuracy of the model
accuracy_MAP = calculate_accuracy(predictions_MAP, num_samples_50)
print('The accuracy of the MAP classifier is: ', accuracy_MAP*100, '%')

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Using the original generated samples without noise
# Calculate the sample mean vector
mean_c1 = np.round(np.mean(generated_samples_c1_100, axis=0),2) # Class 1
mean_c2 = np.round(np.mean(generated_samples_c2_100, axis=0),2) # Class 2

# Calculate the sample covariance matrix
cov_c1 = np.round(np.cov(generated_samples_c1_100, rowvar = False),2) # Class 1
cov_c2 = np.round(np.cov(generated_samples_c2_100, rowvar = False),2) # Class 2

inv_cov_c1 = np.round(np.linalg.inv(cov_c1),2) # Class 1
inv_cov_c2= np.round(np.linalg.inv(cov_c2),2) # Class 2

# Call ML classifier function to classify noisy samples using the training data of 100 samples with no noise
predictions_ML_2 = np.array([ML_classifier(x, mean_c1, mean_c2, cov_c1, cov_c2, inv_cov_c1, inv_cov_c2) for x in test_samples])
# Calculate the accuracy of the model
accuracy_ML_2 = calculate_accuracy(predictions_ML_2, num_samples_50)
print('The accuracy of the ML classifier (for training data without noise) is: ', accuracy_ML_2*100, '%')

# Call MAP classifier function to classify noisy samples using the training data of 100 samples with no noise
predictions_MAP_2 = np.array([MAP_classifier(x, mean_c1, mean_c2, cov_c1, cov_c2, inv_cov_c1, inv_cov_c2, p_c1, p_c2) for x in test_samples])

# Calculate the accuracy of the model
accuracy_MAP_2 = calculate_accuracy(predictions_MAP_2, num_samples_50)
print('The accuracy of the MAP classifier (for training data without noise) is: ', accuracy_MAP_2*100, '%')
