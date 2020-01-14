import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self):
        pass

    def plot_image(self, image):
        # plt.imshow(image,cmap='gray') # See in grayscale
        plt.imshow(image)  # See in grayscale
        plt.show()

    def plotgraph(self, epochs, acc, val_acc):
        """
        Plot training & validation accuracy values
        :param epochs: number of epochs
        :param acc:
        :param val_acc:
        """
        plt.plot(epochs, acc, 'b')
        plt.plot(epochs, val_acc, 'r')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()
