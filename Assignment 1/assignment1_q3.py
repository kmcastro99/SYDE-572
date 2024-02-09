import numpy as np
import matplotlib.pyplot as plt

"""
SYDE 572 - Assignment 1

Author: Karla Castro

Question 3

"""

def generate_random_samples(num_samples, mean, covariance):
    """
    Generates random samples given the mean and covariance
    """
    # Create empty matrix where the generated samples will be saved
    np.random.seed(30)
    generated_samples = np.round(np.random.multivariate_normal(mean, covariance, size = num_samples),2)
    # Calculate the sample mean vector (for 2d)
    mean_vector = np.round(generated_samples.mean(axis=0),2)

    return generated_samples


# Calculate the euclidean distance of two points
def euclidean_distance(x, y):
    """
    Calculates the euclidean distanc of two points
    """
    distance = np.sqrt(np.sum((x - y) ** 2, axis=1))
    return distance


# Define the k-nearest classifier
def knn_classifier(train_set, labels_set, test_set, k):
    """
    Determine the predicted class label for the test data based on the majority class samples
    among its k nearest neighbors
    """
    # Calculate the distance between test_data and the train_data
    euclidean_distances = euclidean_distance(train_set, test_set)
    # Arrange the distances from the smallest to the largest
    arranged_distances = np.argsort(euclidean_distances)
    nearest_points = arranged_distances[:k]
    # Define counters for the class labels
    count_c1 = 0
    count_c2 = 0
    # Count the class occurences
    for i in nearest_points:
        if labels_set[i] == 0:
            count_c1 += 1
        elif labels_set[i] == 1:
            count_c2 += 1
    # Predict the class label
    if count_c1 > count_c2:
        return 0 # Class 1
    else:
        return 1 # Class 2


# Create a grid of points 
def plot_knn(train_set, label_set, k_value):
    # Create points for grid with min and max values
    x_min, x_max = train_set[:, 0].min() - 0.07, train_set[:, 0].max() + 0.07
    y_min, y_max = train_set[:, 1].min() - 0.07, train_set[:, 1].max() + 0.07
    
    # Create a grid points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),np.linspace(y_min, y_max, 500))
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    # Preddict points/labels
    predicted_points = np.array([knn_classifier(train_set, label_set, x_test, k_value) for x_test in grid_points])
    predicted_points_n = predicted_points.reshape(xx.shape)

    plt.contourf(xx, yy, predicted_points_n, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(train_set[label_set == 0, 0], train_set[label_set == 0, 1], c='r', marker='o', label='Class 1')
    plt.scatter(train_set[label_set == 1, 0], train_set[label_set == 1, 1], c='b', marker='o', label='Class 2')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title(f'kNN Classifier k = {k_value}')
    plt.grid(True)
    
    plt.show()


def add_noise(samples):
    mean = [0,0]
    covariance_matrix = np.identity(2)
    white_noise = np.round(np.random.multivariate_normal(mean,covariance_matrix,50),2)
    # Add white noise to the samples
    noisy_samples = samples + white_noise
    return noisy_samples


def calculate_accuracy (train_set, label_set, Test_set, k_value):

    # Use kNN classifier to predict labels
    predicted_labels = np.array([knn_classifier(train_set, label_set, test_set, k_value) for test_set in Test_set ])
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
    return error_classifier


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
num_samples = 100

# Find generated samples, their mean, covariance, eigen values and eigen vectors for C1
generated_samples_c1 = generate_random_samples(num_samples, mean_vector_1, covariance_matrix_1)

# Find generated samples, their mean, covariance, eigen values and eigen vectors for C2
generated_samples_c2 = generate_random_samples(num_samples, mean_vector_2, covariance_matrix_2)

# Join train_data composed of the generated samples for the two classes
train_data = np.vstack((generated_samples_c1, generated_samples_c2))

# Create labels for two classes (Class 1 as 0 and Class 2 as 1)
labels = np.hstack((np.zeros(num_samples), np.ones(num_samples)))

# Compute for k between 1 and 5
k_array = [1, 2, 3, 4 ,5]

# Plot kNN for each k 
for k in range (len(k_array)):
    plot_knn(train_data, labels, k_array[k])


# CALCULATE ACCURACY

# Generate 50 new samples that will have noise
new_samples_c1 = generate_random_samples(50, mean_vector_1, covariance_matrix_1)
new_samples_c2 = generate_random_samples(50, mean_vector_2, covariance_matrix_2)

# Add noise to the samples

noisy_samples_c1 = add_noise(new_samples_c1)
noisy_samples_c2 = add_noise(new_samples_c2)

# Stack noisy samples so that C1 comes before C2
noisy_samples = np.vstack((noisy_samples_c1, noisy_samples_c2))
noisy_labels = np.hstack((np.zeros(50), np.ones(50)))

# Calculate accuracy for k
# Create list to append accuracy values from for loop
accuracy = []
for k in range (len(k_array)):
    # Calculate accuracy for specific k value
    accuracy_values = np.round(calculate_accuracy(train_data, labels, noisy_samples, k_array[k])*100,2)
    accuracy.append(accuracy_values)
    print(f"The accuracy for the kNN classifier with k= {k_array[k]} is:", accuracy_values, "%")

# Plot Accuracy vs K
plt.plot(k_array, accuracy, marker='o', color = 'r')
plt.xlabel("k values")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs k value")
plt.show()