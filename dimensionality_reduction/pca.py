##########################################################################################################
# Implemetation of PCA from scratch
#########################################################################################################
# Reminder: Eigne vectors are the ones that just stretch when we apply the linear transformation by the matrix
# AV = λV 
# where A is the matrix, V is the eigen vector and λ is the eigen value
# The eigen vectors are the directions of the data and the eigen values are the amount of variance in that direction
# (A-λI)V = 0
# where I is the identity matrix, and only holds if the determinant of the matrix (A-λI) is 0
#######################################################################################################
# PCA Steps: 
# 1. Standardize the data (If you do PCA without standardization, features with larger magnitude dominate the result.)
# 2. Compute the covariance matrix
# 3. Compute the eigen values and eigen vectors of the covariance matrix
# 4. Sort the eigen values and eigen vectors
# 5. Select the top k eigen vectors
# 6. Transform the data using the top k eigen vector
##########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


load_iris = load_iris()
X = load_iris.data
y = load_iris.target
# plot the data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris Dataset')
plt.colorbar()
plt.show()

# Standardize the data along each feature
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_std = (X - mean) / std

# Compute the covariance matrix
cov_matrix = np.cov(X_std.T)

# Compute the eigen values and eigen vectors of the covariance matrix
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

# Sort the eigen values and eigen vectors
sorted_indices = np.argsort(eigen_values)[::-1]
sorted_eigen_values = eigen_values[sorted_indices]
sorted_eigen_vectors = eigen_vectors[:, sorted_indices]

# Select the top k eigen vectors
k = 2
eigen_vector_subset = sorted_eigen_vectors[:, :k]
# Transform the data using the top k eigen vector
X_pca = X_std.dot(eigen_vector_subset)
# Plot the data
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.colorbar()
plt.show()

