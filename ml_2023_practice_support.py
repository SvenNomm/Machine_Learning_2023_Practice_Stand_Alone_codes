# this file containes uncategorized support functions
import numpy as np

def generate_circle(center, radius, number_of_elements):
    theta = np.linspace(0, 2 * np.pi, number_of_elements)
    circle = np.zeros((number_of_elements, 2))
    for i in range(0, number_of_elements):
        circle[i, 0] = radius * np.cos(theta[i]) + center[0]
        circle[i, 1] = radius * np.sin(theta[i]) + center[1]

    return circle