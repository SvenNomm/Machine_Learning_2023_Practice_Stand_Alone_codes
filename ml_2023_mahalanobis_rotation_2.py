
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
import scipy
import time
import ml_2023_practice_support as mlps
import ml_2023_my_distance_functions as my_dist


number_of_points = 50
x = np.linspace(-1.5, 1.5, number_of_points)
y = np.linspace(-1, 3, number_of_points)
X,Y = np.meshgrid(x,y)
# compute distances from come selected point (in our case it is center) to all other points using defined metrics
n, m = X.shape
center = np.array([0, 0])
mahalanobis_field = np.zeros((n,m))

center = np.array([0, 0])  # center (centroid)
w = np.array([[0.1, 0], [0, 0.1]])  # eigenvalues
v = np.array([[0.5, 0.2], [0.2, 0.5]])  # eigenvectors

set_1 = mlps.gaussian_generator(center, eig_val=w, eig_vec=v, size=100)


mahalanobis_field = mlps.generate_mahalanobis_field(X, Y, set_1, center)


max_value = np.max(mahalanobis_field)

scales = np.linspace(0,max_value, 100)


fig, axs = plt.subplots(nrows=1,ncols=1, constrained_layout=True, sharey=True)
norm = plt.Normalize(scales.min(), scales.max())
cm = plt.cm.get_cmap('turbo')
set = np.transpose(np.array([X.ravel(), Y.ravel()]))
set_2 = axs.scatter(set[:, 0], set[:, 1], c=mahalanobis_field.ravel(), cmap = cm, alpha=0.1)
set_3 = axs.scatter(set_1[:, 0], set_1[:, 1], facecolors='lightcyan', edgecolors='skyblue')
axs.set_title('Mahalanobis')

sm = ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs)
cbar.ax.set_title("scale")

cbar.set_label('Distance')

def animate(i):
    print(i)
    w = np.array([[0.1, 0], [0, 0.1]])
    theta = i * np.pi / 180
    w = w * (np.abs(np.sin(theta) *0.9) + 0.1)
    set_1 = mlps.gaussian_generator(center, eig_val=w, eig_vec=v, size=100)

    mahalanobis_field = mlps.generate_mahalanobis_field(X, Y, set_1, center)


    set_2.set_offsets(mahalanobis_field)
    #set_3.set_offsets(set_1)
    time.sleep(5)

animation = FuncAnimation(fig, animate, interval=10, repeat=False)
axs.axis('equal')
#animation.save('A.gif',fps=10)
plt.show()

