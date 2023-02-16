
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import scipy

import ml_2023_practice_support as mlps
import ml_2023_my_distance_functions as my_dist


#path = 'C:/Users/Sven/Google Drive/Teaching/Machine_Learning_2023/practice_01/data_01/'
path = '/Users/svennomm/Library/CloudStorage/GoogleDrive-sven.nomm@gmail.com/My Drive/Teaching/Machine_Learning_2023/practice_01/data_01/'
fname = path + 'one_gaussian_2.pkl'
file_handle = open(fname, 'rb')
set_1 = pkl.load(file_handle)


center = np.array([0,0])
radius = 0.5
number_of_elements = 80

circle = mlps.generate_circle(center, radius, number_of_elements)
thetat = np.linspace(0, 2 * np.pi, number_of_elements)

mahalanobis_circle = np.zeros((number_of_elements, 2))


for j in range(0, number_of_elements):
    mahalanobis_circle[j, 0] = my_dist.my_mahalanobis_distance(center, circle[j, :], set_1) * np.cos(thetat[j]) + center[0]
    mahalanobis_circle[j, 1] = my_dist.my_mahalanobis_distance(center, circle[j, :], set_1) * np.sin(thetat[j]) + center[1]

fig, axs = plt.subplots()

set = axs.scatter(set_1[:, 0], set_1[:, 1], c='tab:pink')
mc, = axs.plot(mahalanobis_circle[:, 0], mahalanobis_circle[:, 1], c='darkviolet')
axs.plot(circle[:, 0], circle[:, 1], c='blue', linewidth=5)
axs.scatter(circle[1, 0], circle[1, 1], c='black', s=100)


def animate(i):
    theta = i * np.pi / 180
    rotation_matrix = np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    set_2 = np.matmul(set_1, rotation_matrix)
    mahalanobis_circle_2 = np.matmul(mahalanobis_circle, rotation_matrix)
    set.set_offsets(set_2)

    mc.set_data(mahalanobis_circle_2[:, 0], mahalanobis_circle_2[:, 1])

animation = FuncAnimation(fig, animate, interval=10, repeat=False)
axs.axis('equal')
animation.save('A.gif',fps=10)
plt.show()

