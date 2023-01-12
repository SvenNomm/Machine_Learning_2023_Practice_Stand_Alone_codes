# this file containes uncategorized support functions
import numpy as np
import ml_2023_my_distance_functions as my_dist

def generate_circle(center, radius, number_of_elements):
    theta = np.linspace(0, 2 * np.pi, number_of_elements)
    circle = np.zeros((number_of_elements, 2))
    for i in range(0, number_of_elements):
        circle[i, 0] = radius * np.cos(theta[i]) + center[0]
        circle[i, 1] = radius * np.sin(theta[i]) + center[1]

    return circle


def gaussian_generator(center, eig_val, eig_vec, size):
    sigma = np.matmul(np.matmul(eig_vec, eig_val), np.transpose(eig_vec))
    set = np.random.multivariate_normal(center, sigma, size)

    return set


def generate_mahalanobis_field(X, Y, set, center):
    n, m = X.shape
    mahalanobis_field = np.zeros((n, m))
    for i in range(0, n):
        for j in range(0, m):
            point_of_interest = np.array([X[i, j], Y[i, j]])
            mahalanobis_field[i, j] = my_dist.my_mahalanobis_distance(center, point_of_interest, set)

    return mahalanobis_field

