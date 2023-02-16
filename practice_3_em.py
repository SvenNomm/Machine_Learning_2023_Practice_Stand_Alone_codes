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
print(sys.version)
from IPython.display import clear_output
from practice_3_functions import *

#path = 'C:/Users/Sven/Google Drive/Teaching/Machine_Learning_2023/practice_02/data_02/'
path = '/Users/svennomm/Library/CloudStorage/GoogleDrive-sven.nomm@gmail.com/My Drive/Teaching/Machine_Learning_2023/practice_02/data_02/'
mpl.rc("figure", dpi=150)

fname = path + 'data_set_2D.pkl'
file_handle = open(fname, 'rb')
data_set_3 = pkl.load(file_handle)


number_of_clusters = 3
n, m = data_set_3.shape
sigma_hat = {}

tau = np.ones(number_of_clusters) / 3
orient=np.array([[4,2],[2,3]])
radiuses = np.array([[3, 0],[0, 2]])
mu_hat = np.random.normal(0, 1, size=(number_of_clusters, 2))

for k in range (0, number_of_clusters):
    #a = np.array([[1,3],[4,2]])
    sigma_hat[k] = np.matmul(np.matmul(orient, radiuses), np.transpose(orient))
    tau[k] = np.random.uniform(0, 1)

fig_2 = plt.figure(figsize=(2,2))

flag = 0

colour_set = ['r', 'g', 'brown']

loop_nr = 0
print('hello')

while flag==0:
    plt.clf()
    plt.scatter(data_set_3[:, 0], data_set_3[:, 1], c='b')
    print('hello')
    for k in range(0, number_of_clusters):
        print(loop_nr)

        eigenvalues, eigenvectors = np.linalg.eig(sigma_hat[k])
        #eigenvalues = eigenvalues * 0.9
        eigenvalues[0] = np.sqrt(eigenvalues[0]) * 2
        eigenvalues[1] = np.sqrt(eigenvalues[1]) * 2
        ellipse =my_ellipse(eigenvalues, mu_hat[k, :], np.transpose(eigenvectors), 20)
        plt.plot(ellipse[:, 0], ellipse[:, 1], c=colour_set[k])
        plt.scatter(mu_hat[k,0], mu_hat[k, 1], c=colour_set[k])
    plt.show()

    r = my_e_step(data_set_3, tau, sigma_hat, mu_hat)
    idx = np.argmax(r, axis=1)
    mu_hat, sigma_hat, tau = my_m_step(data_set_3, r)
    loop_nr = loop_nr + 1
    if loop_nr > 100:
        flag = 1

plt.show()