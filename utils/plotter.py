import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self):
        pass

    @staticmethod
    def plot_image( image):
        # plt.imshow(image,cmap='gray') # See in grayscale
        plt.imshow(image)  # See in grayscale
        plt.show()

    @staticmethod
    def plotgraph(title, epochs, acc, val_acc):
        """
        Plot training & validation accuracy values
        :param epochs: number of epochs
        :param acc:
        :param val_acc:
        """
        plt.plot(epochs, acc, 'b')
        plt.plot(epochs, val_acc, 'r')
        plt.ylim((0, 1))
        plt.title(title + ' accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

    @staticmethod
    def plotlossgraph(title, epochs, loss, val_loss):
        """
        Plot training & validation accuracy values
        :param epochs: number of epochs
        :param acc:
        :param val_acc:
        """
        plt.plot(epochs, loss, 'b')
        plt.plot(epochs, val_loss, 'r')
        plt.title(title + ' loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()
