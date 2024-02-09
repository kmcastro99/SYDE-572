import numpy as np
import matplotlib.pyplot as plt

"""
SYDE 572 - Assignment 2

Author: Karla Castro

Question 1

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

# Function to define the values (1,0) for the decision boundary
def MMD_classifier(x, mean_c1, mean_c2, cov_inv_c1, cov_inv_c2):

    # Discriminant Function for C1
    g1 = np.dot(np.dot((x - mean_c1).T, cov_inv_c1), (x - mean_c1))

    # Discriminant Function for C2
    g2 = np.dot(np.dot((x - mean_c2).T, cov_inv_c2), (x - mean_c2))

    if g1 < g2:
        return 0 # Label Class 1
    else:
        return 1 # Label Class 2

# Function to determine the predicted values and to plot the decision boundary 
def MDD_plot_boundary(samples_c1, samples_c2, mean_c1, mean_c2, cov_inv_c1, cov_inv_c2, grid_points = True):
        
    # Stack all samples from C1 and C2
    samples = np.vstack((samples_c1, samples_c2))

    # x and y values using our samples classes

    x = [arr[0] for arr in samples]
    y = [arr[1] for arr in samples]

    # Find the minimum and maximum values for x and y
    x_min, x_max = np.min(x) - 1, np.max(x) + 1
    y_min, y_max = np.min(y) - 1, np.max(y) + 1

    # Create a grid of points
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    x, y = np.meshgrid(x_range, y_range)
    grid_points = np.c_[x.ravel(), y.ravel()]

    # Predictions
    predicted_points = np.array([MMD_classifier(point, mean_c1, mean_c2, cov_inv_c1, cov_inv_c2 ) for point in grid_points])

    # Plot the samples and decision boundary
    plt.scatter([arr[0] for arr in samples_c1], [arr[1] for arr in samples_c1], label='Class 1', marker='.', color='blue')
    plt.scatter([arr[0] for arr in samples_c2], [arr[1] for arr in samples_c2], label='Class 2', marker='x', color='red')
    # Plot new samples with white noise
    plt.scatter(mean_c1[0], mean_c1[1], label = 'Mean Class 1', marker='D', color='k' )
    plt.annotate(mean_c1, mean_c1)
    plt.scatter(mean_c2[0], mean_c2[1], label = 'Mean Class 2', marker='D', color='g' )
    plt.annotate(mean_c2, mean_c2)

    # Plot filled contours
    contour_plot = plt.contourf(x, y, predicted_points.reshape(x.shape), alpha=0.1).collections[1].get_paths()[0].vertices
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title(f'MMD Classifier Decision Boundary with u2 = {mean_vector_2}')
    plt.grid(True)
    plt.show()

    return predicted_points

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

def med_classifier(x, mean_vector_C1, mean_vector_C2):
    """
    Find the labels using the med classifier
    """
    # Discriminant Function for C1 (Euclidean Distance between x and the prototype of C1)
    g1 = np.dot((-mean_vector_C1), x.T) + (1/2)*np.dot(mean_vector_C1, mean_vector_C1.T)

    # Discriminant Function for C2 (Euclidean Distance between x and the prototype of C2)
    g2 = np.dot((-mean_vector_C2), x.T) + (1/2)*np.dot(mean_vector_C2, mean_vector_C2.T)

    if g1 < g2:
        return 0 # Label Class 1
    else:
        return 1 # Label Class 2


# Define the mean vector (for both 5 and 100 samples)
# Class 1
mean_vector_1 = np.array([1, 3])
# Class 2
mean_vector_2 = np.array([4, 7]) # Change to [4, 7] or [20, 31]

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
num_samples_5 = 5 # For question asking for 5 samples
num_samples_100 = 100 # For question asing for 100 samples
num_samples_50 = 50 # For question of 50 samples with white noise

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Calculation for NUM_SAMPLES = 5
# Generate samples for the classes above
generated_samples_c1_5 = generate_random_samples(num_samples_5, mean_vector_1, covariance_matrix_1) # Class 1 samples (5 samples)
generated_samples_c2_5 = generate_random_samples(num_samples_5, mean_vector_2, covariance_matrix_2) # Class 2 samples (5 samples)

# Generate noisy samples to add to each sample from generated_samples_c1 and generated_samples_c2
new_samples_c1_5, new_samples_c2_5 = add_noise(generated_samples_c1_5, generated_samples_c2_5, mean_vector_new, covariance_matrix_new, num_samples_5)
# Print the generated samples
print('Generated samples for class 2:', new_samples_c1_5) # Class 1
print('Generated samples for class 2:', new_samples_c2_5) # Class 2

# Calculate the sample mean vector (for 2d)
mean_vector_c1_5 = np.round(np.mean(new_samples_c1_5, axis=0),2) # Class 1
# print(mean_vector_c1_5)
mean_vector_c2_5 = np.round(np.mean(new_samples_c2_5, axis=0),2) # Class 2
# print(mean_vector_c2_5)

# Calculate the sample covariance matrix
covariance_matrix_c1_5 = np.round(np.cov(new_samples_c1_5, rowvar = False),2) # Class 1
# print(covariance_matrix_c1_5)
covariance_matrix_c2_5 = np.round(np.cov(new_samples_c2_5, rowvar = False),2) # Class 2
# print(covariance_matrix_c2_5)

# Calculate the eigen vectors
eigen_value_c1_5,eigen_vector_c1_5 = np.linalg.eig(covariance_matrix_c1_5) # Class 1
eigen_value_c2_5,eigen_vector_c2_5 = np.linalg.eig(covariance_matrix_c2_5) # Class 2
# print('E-value:', np.round(eigen_value_c1_5,2))
# print('E-vector', np.round(eigen_vector_c1_5,2))
# print('E-value:', np.round(eigen_value_c2_5,2))
# print('E-vector', np.round(eigen_vector_c2_5,2))

# Find the inverse of the covariance matrix
inv_cov_c1_5 = np.round(np.linalg.inv(covariance_matrix_c1_5),2) # Class 1
inv_cov_c2_5 = np.round(np.linalg.inv(covariance_matrix_c2_5),2) # Class 2
# print(inv_cov_c1_5)
# print(inv_cov_c2_5)

# Call the plot boundary function to determine decision boundary using MDD
MDD_plot_boundary(new_samples_c1_5, new_samples_c2_5, mean_vector_c1_5, mean_vector_c2_5, inv_cov_c1_5 , inv_cov_c2_5)

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Calculation for NUM_SAMPLES = 100
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

# Calculate the eigen vectors
eigen_value_c1_100,eigen_vector_c1_100 = np.linalg.eig(covariance_matrix_c1_100) # Class 1
eigen_value_c2_100,eigen_vector_c2_100 = np.linalg.eig(covariance_matrix_c2_100) # Class 2

# Find the inverse of the covariance matrix
inv_cov_c1_100 = np.round(np.linalg.inv(covariance_matrix_c1_100),2) # Class 1
inv_cov_c2_100 = np.round(np.linalg.inv(covariance_matrix_c2_100),2) # Class 2

# Call the plot boundary function to determine decision boundary using MDD
MDD_plot_boundary(new_samples_c1_100, new_samples_c2_100, mean_vector_c1_100, mean_vector_c2_100, inv_cov_c1_100 , inv_cov_c2_100)

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Generate 50 new samples per class and add white noise to each of the 100 samples
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
noise_samples = np.vstack((white_noise_c1_50, white_noise_c2_50))

# Call MDD classifier function to classify samples with MDD trained with 100 samples
predictions_50_MDD100 = np.array([MMD_classifier(point, mean_vector_c1_100, mean_vector_c2_100, inv_cov_c1_100, inv_cov_c2_100 ) for point in noise_samples])
MDD_plot_boundary(white_noise_c1_50, white_noise_c2_50, mean_vector_c1_100, mean_vector_c2_100, inv_cov_c1_100 , inv_cov_c2_100)

# Call MDD classifier function to  to classify samples with MDD trained with 5 samples
predictions_50_MDD5 = np.array([MMD_classifier(point, mean_vector_c1_5, mean_vector_c2_5, inv_cov_c1_5, inv_cov_c2_5 ) for point in noise_samples])
MDD_plot_boundary(white_noise_c1_50, white_noise_c2_50, mean_vector_c1_5, mean_vector_c2_5, inv_cov_c1_5 , inv_cov_c2_5)

# Call function to calculate accuracy
accuracy_MDD100 = np.round(calculate_accuracy (predictions_50_MDD100, num_samples_50),2)
accuracy_MDD5 = np.round(calculate_accuracy (predictions_50_MDD5, num_samples_50),2)
print('The MMD model trainned with 100 samples is:', np.round(accuracy_MDD100*100,2), '%', 'accurate')
print('The MMD model trainned with 5 samples is:', np.round(accuracy_MDD5*100,2), '%', 'accurate')

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Campare the MMD Classifier to the MED Classifier

# Call MED classifier function to classify samples with MED trained with 100 samples
predictions_50_MED100 = np.array([med_classifier(point, mean_vector_c1_100, mean_vector_c2_100) for point in noise_samples])

# Call MED classifier function to classify samples with MED trained with 100 samples
predictions_50_MED5 = np.array([med_classifier(point, mean_vector_c1_5, mean_vector_c2_5) for point in noise_samples])

# Call function to calculate accuracy
accuracy_MED100 = np.round(calculate_accuracy (predictions_50_MED100, num_samples_50),2)
accuracy_MED5 = np.round(calculate_accuracy (predictions_50_MED5, num_samples_50),2)
print('The MED model trainned with 100 samples is:', np.round(accuracy_MED100*100,2), '%', 'accurate')
print('The MED model trainned with 5 samples is:', np.round(accuracy_MED5*100,2), '%', 'accurate')

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------