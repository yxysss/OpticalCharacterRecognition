import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self):
        pass

    def plot_image(self, image):
        # plt.imshow(image,cmap='gray') # See in grayscale
        plt.imshow(image) # See in grayscale
        plt.show()
