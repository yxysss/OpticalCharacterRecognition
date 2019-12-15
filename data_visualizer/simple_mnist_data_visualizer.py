import matplotlib.pyplot as plt
import numpy as np

from utils.plotter import Plotter


class SimpleMnistDataVisualizer:
    def __init__(self, X_train, y_train, mapp):
        self.X_train = X_train
        self.y_train = y_train
        self.mapp = mapp
        self.plotter = Plotter()

    def plot_first_digit(self, ):
        self.plotter.plot_image(self.X_train[0].reshape(28, 28))

    def plot_range(self):
        for i in range(100, 109):
            plt.subplot(330 + (i + 1))
            plt.imshow(self.X_train[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
            plt.title(chr(self.mapp[np.argmax(self.y_train[i])]))
        plt.show()

    def plotgraph(self, epochs, acc, val_acc):
        self.plotter.plotgraph(epochs, acc, val_acc)
