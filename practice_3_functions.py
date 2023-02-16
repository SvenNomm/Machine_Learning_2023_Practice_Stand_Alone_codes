import numpy as np
#%matplotlib widget
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mpl
import pickle as pkl
import time
import os
import random
import sys
from scipy.stats import multivariate_normal



def my_ellipse(radii, mu, rot_mat, steps_nr):
    theta = np.linspace(0, 2* np.pi, steps_nr, endpoint=True)


    ellipse = np.zeros((steps_nr,2))

    for i in range(0, steps_nr):
        ellipse[i, 0] = radii[0] * np.cos(theta[i])
        ellipse[i, 1] = radii[1] * np.sin(theta[i])

    ellipse = np.matmul(ellipse, rot_mat)

    ellipse[:, 0] = ellipse[:, 0] + mu[0]
    ellipse[:, 1] = ellipse[:, 1] + mu[1]

    return ellipse


def my_e_step(data_set, tau, sigma_hat, mu_hat):
    n = len(data_set)
    number_of_clusters = len(tau)
    r = np.zeros((n, number_of_clusters))
    for i in range(0, n):
        numerator = np.zeros(number_of_clusters)
        for k in range(0, number_of_clusters):

            var = multivariate_normal(mean=mu_hat[k, :], cov=sigma_hat[k])
            numerator[k] = tau[k] * var.pdf(data_set[i, :])

        denum = np.sum(numerator)
        for k in range(0, number_of_clusters):
            r[i, k] = numerator[k] / denum

    return r

def my_m_step(data_set, r):
    n, number_of_clusters = r.shape
    tau = np.zeros(number_of_clusters)
    _, m = data_set.shape

    mu_hat = np.zeros((number_of_clusters, m))

    sigma_hat = {}

    for k in range(0, number_of_clusters):
        tau[k] = sum(r[:, k]) / n

        nominator_rx = np.zeros((n,m))
        nominator_rxmu = np.zeros((m,m))

        for i in range(0, n):
            nominator_rx[i, :] = data_set[i, :] * r[i, k]
            ds = data_set[i, :].reshape((1,2))
            nominator_rxmu = nominator_rxmu + np.matmul(np.transpose(ds), ds) * r[i, k]

        mu_hat[k, :] = np.sum(nominator_rx, axis=0) /np.sum(r[:, k])

        sigma_hat[k] = (nominator_rxmu / np.sum(r[:, k])) - np.matmul((mu_hat[k, :].reshape((2,1))), mu_hat[k, :].reshape((1,2)))

    return mu_hat, sigma_hat, tau

