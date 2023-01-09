# This is script to generate data sets

import numpy as np
import pickle as pkl

import pandas as pd
import os
import matplotlib.pyplot as plt


path = 'C:/Users/Sven/Google Drive/Teaching/Machine_Learning_2023/practice_01/data_01/'

n_1 = 200  # number of the elements
mu_1 = np.array([0, 1])  # center (centroid)

# sigma is expected to be positive  semi-definite matrix
w = np.array([[0.4, 0], [0, 0.4]])  # eigenvalues

v = np.array([[0.5, 0.2], [0.2, 0.5]])  # eigenvectors

sigma = np.matmul(np.matmul(v, w), np.transpose(v))

# verify if the matrix is positive semidefinite
# print(np.all(np.linalg.eigvals(sigma) > 0))

set_1 = np.random.multivariate_normal(mu_1, sigma, n_1)

fname = path + 'one_gaussian.pkl'

file_handle = open(fname,"wb")
pkl.dump(set_1,  file_handle)

fig = plt.figure()
s = plt.scatter(set_1[:, 0], set_1[:, 1])
plt.show()

fig_fname = path + 'one_gaussian.pdf'
fig.savefig(fig_fname)

# saving into other formats
df_set_1 = pd.DataFrame(set_1)

excel_name = path + 'one_gaussian.xlsx'
df_set_1.to_excel(excel_name, index=False, header=False)

csv_name = path + 'one_gaussian.csv'
df_set_1.to_csv(csv_name, index=False, header=False)

print("That's all folks!!!")
