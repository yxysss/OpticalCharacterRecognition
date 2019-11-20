import matplotlib.pyplot as plt
import numpy as np

from utils.plotter import Plotter


class SimpleMnistDataVisualizer:
    def __init__(self, data):
        self.data = data
        self.plotter = Plotter()

    def plot_first_digit(self,):
        self.plotter.plot_image(self.data[0].reshape(28,28))
