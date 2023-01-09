
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import scipy


#path = 'C:/Users/Sven/Google Drive/Teaching/Machine_Learning_2023/practice_01/data_01/'
path = '/Users/svennomm/Library/CloudStorage/GoogleDrive-sven.nomm@gmail.com/My Drive/Teaching/Machine_Learning_2023/practice_01/data_01/'
fname = path + 'one_gaussian.pkl'
file_handle = open(fname, 'rb')
set_1 = pkl.load(file_handle)


fig_3, axs_3 = plt.subplots()

set = axs_3.scatter(set_1[:, 0], set_1[:, 1], c='tab:pink')


def animate(i):
    theta = i * np.pi / 180
    rotation_matrix = np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    set_2 = np.matmul(set_1, rotation_matrix)
    set.set_offsets(set_2)


animation = FuncAnimation(fig_3, animate, interval=10, repeat=False)
animation.save('A.gif',fps=10)
plt.show()

