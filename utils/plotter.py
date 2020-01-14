import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self):
        pass

    def plot_image(self, image):
        # plt.imshow(image,cmap='gray') # See in grayscale
        plt.imshow(image)  # See in grayscale
        plt.show()

    def plotgraph(self, epochs, rate, val_rate, ylabel):
        # Plot training & validation accuracy values
        plt.plot(epochs, rate, 'b')
        plt.plot(epochs, val_rate, 'r')
        plt.title("Model " + ylabel)
        plt.ylabel(ylabel)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()
