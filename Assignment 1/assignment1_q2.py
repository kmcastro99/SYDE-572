import numpy as np
import matplotlib.pyplot as plt

"""
SYDE 572 - Assignment 1

Author: Karla Castro

Question 2

"""

def generate_random_samples(num_samples, mean, covariance):
    """
    Generates random samples given the mean and covariance
    """
    # Create empty matrix where the generated samples will be saved
    np.random.seed(30)
    generated_samples = np.round(np.random.multivariate_normal(mean, covariance, size = num_samples),2)
    # Calculate the sample mean vector
    mean_vector = np.round(generated_samples.mean(axis=0),2)

    return generated_samples, mean_vector

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

# Define the mean vector
# Class 1
mean_vector_1 = np.array([4,7])
# Class 2
mean_vector_2 = np.array([5,10])

# Define the summation matrix
# Class 1
covariance_matrix_1 = np.array([[9,3],[3,10]])
# Class 2
covariance_matrix_2 = np.array([[7,0],[0,16]])

# Define number of samples
num_samples = 100 # Change to 5 to check for 5 samples

# Find generated samples, their mean, covariance, eigen values and eigen vectors for C1
generated_samples_c1, mean_vector_c1 = generate_random_samples(num_samples, mean_vector_1, covariance_matrix_1)

# Find generated samples, their mean, covariance, eigen values and eigen vectors for C2
generated_samples_c2, mean_vector_c2 = generate_random_samples(num_samples, mean_vector_2, covariance_matrix_2)

# Create a grid of points to plot the decision boundary
x, y = np.meshgrid(np.linspace(-3, 12, 1000), np.linspace(-3, 25, 1000))
grid_points = np.c_[x.ravel(), y.ravel()]

# Predictions
predicted_points = np.array([med_classifier(point, mean_vector_c1, mean_vector_c2) for point in grid_points])

# Plot the samples and decision boundary
plt.scatter(generated_samples_c1[:, 0], generated_samples_c1[:, 1], label='Class 1', marker='.', color='blue')
plt.scatter(generated_samples_c2[:, 0], generated_samples_c2[:, 1], label='Class 2', marker='x', color='red')
# Plot new samples with white noise
plt.scatter(mean_vector_c1[0], mean_vector_c1[1], label = 'Mean Class 1', marker='D', color='k' )
plt.annotate(mean_vector_c1, mean_vector_c1)
plt.scatter(mean_vector_c2[0], mean_vector_c2[1], label = 'Mean Class 2', marker='D', color='g' )
plt.annotate(mean_vector_c2, mean_vector_c2)

# Plot filled contours
contour_plot = plt.contourf(x, y, predicted_points.reshape(x.shape), alpha=0.1).collections[1].get_paths()[0].vertices
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Minimum Euclidean Distance (MED) Classifier')
plt.grid(True)
plt.show()

# Print Decision Boundary Equation
coefficients_equation = np.polyfit(contour_plot[:,0], contour_plot[:,1], 1)
line_equation = f"y = {coefficients_equation[0]:.2f}x + {coefficients_equation[1]:.2f}"
slope = np.round(coefficients_equation[0],2)
offset = np.round(coefficients_equation[1],2)
print(f"The Decision Boundary Equation is a straight line with a slope of {slope} and an offset of {offset} ")
print("The line equation is:", line_equation)


# QUESTION 2 - PART 3

"""
Report the MED classifier accuracy given 50 samples with noise for both Class 1 and Class 2
"""

# Generate 50 new samples that will have noise
new_samples_c1, mean_noise_c1 = generate_random_samples(50, mean_vector_1, covariance_matrix_1)
new_samples_c2, mean_noise_c2 = generate_random_samples(50, mean_vector_2, covariance_matrix_2)

# Create function to add noise to the samples
def add_noise(samples):
    mean = [0,0]
    covariance_matrix = np.identity(2)
    white_noise = np.round(np.random.multivariate_normal(mean,covariance_matrix,50),2)
    # Add white noise to the samples
    noisy_samples = samples + white_noise
    return noisy_samples

noisy_samples_c1 = add_noise(new_samples_c1)
noisy_samples_c2 = add_noise(new_samples_c2)
# Stack noisy samples so that C1 comes before C2
noisy_samples = np.vstack((noisy_samples_c1, noisy_samples_c2))


# Using 100 samples classifier
# Predict the labels for the new noisy samples using the med_classifier
predicted_labels = np.array([med_classifier(point, mean_vector_c1, mean_vector_c2) for point in noisy_samples])
print('The predicted labels using MED for 100 samples are:', predicted_labels)

# Counter for correct predictions
count_correct_predictions = 0

# Check if the first 50 labels belong to C1
for point in predicted_labels[:50]:
    if point == 0:
        # Add to counter if a label belongs to C1
        count_correct_predictions += 1
# Check if the last 50 labels belong to C2
for point in predicted_labels[-50:]:
    # Add to counter if a label belongs to C2
    if point == 1:
        count_correct_predictions += 1

# Report the classification accuracy
total_data_test = 100
error_classifier = count_correct_predictions/total_data_test

print("The error for the MED classifier trained with 100 samples is:", error_classifier)


# Generate 5 samples
# Find generated samples, their mean, covariance, eigen values and eigen vectors for C1
generated_samples_c1_5, mean_vector_c1_5 = generate_random_samples(5, mean_vector_1, covariance_matrix_1)
# Find generated samples, their mean, covariance, eigen values and eigen vectors for C2
generated_samples_c2_5, mean_vector_c2_5 = generate_random_samples(5, mean_vector_2, covariance_matrix_2)

# Usign 5 samples classifier
# Predict the labels for the new noisy samples using the med_classifier
predicted_labels_5 = np.array([med_classifier(point, mean_vector_c1_5, mean_vector_c2_5) for point in noisy_samples])
print('The predicted labels using MED for 5 samples are:', predicted_labels)

# Counter for correct predictions
count_correct_predictions_MED_5 = 0

# Check if the first 50 labels belong to C1
for point in predicted_labels_5[:50]:
    if point == 0:
        # Add to counter if a label belongs to C1
        count_correct_predictions_MED_5 += 1
# Check if the last 50 labels belong to C2
for point in predicted_labels_5[-50:]:
    # Add to counter if a label belongs to C2
    if point == 1:
        count_correct_predictions_MED_5 += 1

# Report the classification accuracy
total_data_test = 100
error_classifier = count_correct_predictions_MED_5/total_data_test
print("The error for the MED classifier trained with 5 samples is:", error_classifier)









