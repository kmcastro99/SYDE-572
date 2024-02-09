import numpy as np
import matplotlib.pyplot as plt

"""
SYDE 572 - Assignment 1

Author: Karla Castro

Question 1

"""

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

def generate_random_samples(num_samples, mean, covariance):
    """
    Generates random samples given the mean and covariance
    """
    # Create empty matrix where the generated samples will be saved
    np.random.seed(30)
    generated_samples = np.round(np.random.multivariate_normal(mean, covariance, size = num_samples),2)
    # Calculate the sample mean vector (for 2d)
    mean_vector = np.round(generated_samples.mean(axis=0),2)
    # Calculate the covariance matrix
    covariance_matrix = np.round(np.cov(generated_samples, rowvar = False),2)
    # Calculate eigen vectors and eigen values
    eigen_value,eigen_vector = np.linalg.eig(covariance_matrix)

    return generated_samples, mean_vector, covariance_matrix, eigen_value, eigen_vector

#Plot the equiprobability countours for the two classes, follwoing tutorial example
def plot_equiprobability_contour(samples, mean, covariance):
    #Create grid points to evaluate thee Gaussian PDF
    
    x, y = np.meshgrid(np.linspace(-5,17,500), np.linspace(-5,25,500))
    pos = np.dstack((x,y))

    #Define the Gaussian PDF Function

    def gaussian_pdf(x):
        n = mean.shape[0]
        det_cov = np.linalg.det(covariance)
        inv_cov = np.linalg.inv(covariance)
        diff = x-mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=2)
        return(1.0/(2*np.pi*np.sqrt(det_cov)))*np.exp(exponent)
    #Find Gaussian PDF values
    pdf_values = gaussian_pdf(pos)

    #Create scatter plot of the generated data points
    plt.scatter(samples[:,0], samples[:,1], marker= "X", alpha=0.8, label='Generated Samples')

    #Create equiprobability countor
    countor_plot = plt.contourf(x, y, pdf_values, levels=100, cmap='viridis', alpha=0.4)
    return countor_plot

#Find generated samples, their mean, covariance, eigen values and eigen vectors for C1
generated_samples_c1, mean_vector_c1, covariance_matrix_c1, eigen_value_c1, eigen_vector_c1 = generate_random_samples(num_samples, mean_vector_1, covariance_matrix_1)

print("Class 1 Results:")
print("The generated samples are:", generated_samples_c1)
print("Sample Mean:", mean_vector_c1)
print("Sample Covariance:", covariance_matrix_c1),
print('E-value:', eigen_value_c1)
print('E-vector', eigen_vector_c1)

#Find generated samples, their mean, covariance, eigen values and eigen vectors for C2
generated_samples_c2, mean_vector_c2, covariance_matrix_c2, eigen_value_c2, eigen_vector_c2 = generate_random_samples(num_samples, mean_vector_2, covariance_matrix_2)


print("Class 2 Results:")
print("The generated samples are:", generated_samples_c2)
print("Sample Mean:", mean_vector_c2)
print("Sample Covariance:", covariance_matrix_c2),
print('E-value:', eigen_value_c2)
print('E-vector', eigen_vector_c2)

#Create Countor Plot for Class 1
countor_plot_1 = plot_equiprobability_contour(generated_samples_c1, mean_vector_c1, covariance_matrix_c1)
plt.colorbar(countor_plot_1, label='PDF Value')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Equiprobability Contours Class 1')
plt.legend()
plt.grid(True)
#Display plot
plt.show()

#Create Countor Plot for Class 2
countor_plot_2 = plot_equiprobability_contour(generated_samples_c2, mean_vector_c2, covariance_matrix_c2)
plt.colorbar(countor_plot_2, label='PDF Value')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Equiprobability Contours Class 2')
plt.legend()
plt.grid(True)
#Display plot
plt.show()