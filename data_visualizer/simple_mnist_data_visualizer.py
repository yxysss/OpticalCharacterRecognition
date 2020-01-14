import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plotter import Plotter


class SimpleMnistDataVisualizer:
    def __init__(self, X_train, y_train, mapp):
        self.X_train = X_train
        self.y_train = y_train
        self.mapp = mapp
        self.plotter = Plotter()

    def plot_specified_digit(self, code):
        for i in range(self.y_train.shape[0]):
            if (self.y_train[i, code] == 1):
                self.plotter.plot_image(self.X_train[i].reshape(28,28))
                break

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

    def plot_loss_acc(self):
        history_path = './experiments/2019-12-25/xy_emnist_from_config/history_params/parameters.csv'
        history_path = './experiments/2019-12-25/xy_emnist_x_from_config/history_params/parameters.csv'
        data_frame = pd.read_csv(history_path, delimiter=',')
        self.plotter.plotgraph(data_frame['epoch'].values, data_frame['acc'].values, data_frame['val_acc'], "Accuracy")
        self.plotter.plotgraph(data_frame['epoch'].values, data_frame['loss'].values, data_frame['val_loss'], "Loss")

