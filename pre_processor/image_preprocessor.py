import numpy as np
import cv2
from PIL import Image
from utils.plotter import Plotter


class ImagePreProcessor():

    def __init__(self):
        self.plotter = Plotter()

    def execute(self, image):
        # create an array where we can store our picture
        images = np.zeros((1, 784))
        # and the correct values
        correct_vals = np.zeros((1, 10))
        # read the image and transform to black and white
        gray = cv2.imread(image, 0)
        self.plotter.plot_image(gray)
        # resize the images and invert it (black background)
        gray = cv2.resize(255 - gray, (28, 28))
        self.plotter.plot_image(gray)
        cv2.imwrite("test_images/image_1.png", gray)

        """
        all images in the training set have an range from 0-1
        and not from 0-255 so we divide our flatten images
        (a one dimensional vector with our 784 pixels)
        to use the same 0-1 based range
        """
        # im = gray / 255
        im = gray
        return im
