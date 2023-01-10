import numpy as np

def my_euclidean_distance(point_1, point_2):
    dist = 0
    for i in range(0, len(point_1)):
        dist = dist + (point_1[i]-point_2[i]) ** 2

    return np.sqrt(dist)

def my_manhattan_distance(point_1, point_2):
    dist = 0
    for i in range(0, len(point_1)):
        dist =dist + np.abs(point_1[i]-point_2[i])

    return dist

def my_chebyshev_distance(point_1, point_2):
    dist = np.max(np.abs(point_1 - point_2))

    return dist


def my_canberra_distance(point_1, point_2):
    dist = 0
    nr_zeros = 0
    for i in range(0, len(point_1)):
        dist = dist + (np.abs(point_1[i] - point_2[i])) / (np.abs(point_1[i]) + np.abs(point_2[i]))
        if point_1[i] == 0 and point_2[i] == 0:
            nr_zeros = nr_zeros + 1

        if not nr_zeros == 0:
            dist = dist/nr_zeros
    return dist


def my_mahalanobis_distance(point_1, point_2, underling_data_set):
    dist = np.sqrt(np.matmul(np.matmul((point_1-point_2), np.linalg.inv(np.cov(np.transpose(underling_data_set)))), np.transpose(point_1 - point_2)))
    return dist
