import math

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

from constants import HEIGHT, WIDTH
from utils.plotter import Plotter
import matplotlib.pyplot as plt


class ImagePreProcessor():

    def __init__(self):
        self.plotter = Plotter()

    def execute(self, image, readImage=True):
        """
        Pre process image to make it look as close as possible to emnist data set
        :param image:
        :param readImage:
        :return:
        """
        # create an array where we can store our picture
        images = np.zeros((1, 784))
        # and the correct values
        correct_vals = np.zeros((1, 10))
        # read the image and transform to black and white
        gray = 0
        if readImage:
            gray = cv2.imread(image, 0)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            (threshi, img_bw) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.plotter.plot_image(gray) # Plot image in gray scale
        gray = self.cut(gray) # Cut image
        if len(gray) > 0:
            # resize the images and invert it (black background)
            gray = cv2.resize(255 - gray, (28, 28))
            self.plotter.plot_image(gray) # Plot cut image
            cv2.imwrite("test_images/image_1.png", gray) # Save image for testing purposes
            # better black and white version
            (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            self.plotter.plot_image(gray)

            """
            We want to fit the images into this 20x20 pixel box. 
            Therefore we need to remove every row and column at the sides 
            of the image which are completely black.
            """
            while np.sum(gray[0]) == 0:
                gray = gray[1:]

            while np.sum(gray[:, 0]) == 0:
                gray = np.delete(gray, 0, 1)

            while np.sum(gray[-1]) == 0:
                gray = gray[:-1]

            while np.sum(gray[:, -1]) == 0:
                gray = np.delete(gray, -1, 1)

            rows, cols = gray.shape

            """
            Now we want to resize our outer box to fit it into a 20x20 box. 
            We need a resize factor for this.
            """
            if rows > cols:
                factor = 20.0 / rows
                rows = 20
                cols = int(round(cols * factor))
                # first cols than rows
                gray = cv2.resize(gray, (cols, rows))
            else:
                factor = 20.0 / cols
                cols = 20
                rows = int(round(rows * factor))
                # first cols than rows
                gray = cv2.resize(gray, (cols, rows))
            """
            But at the end we need a 28x28 pixel image so we add the missing 
            black rows and columns using the np.lib.pad function which adds 0s 
            to the sides.
            """
            colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
            rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
            gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')
            self.plotter.plot_image(gray)

            shiftx, shifty = self.get_best_shift(gray)
            shifted = self.shift(gray, shiftx, shifty)
            gray = shifted

            cv2.imwrite("test_images/image_2.png", gray)

            """
            all images in the training set have an range from 0-1
            and not from 0-255 so we divide our flatten images
            (a one dimensional vector with our 784 pixels)
            to use the same 0-1 based range
            """
            # im = gray / 255
        im = gray
        return im

    def cut(self, img):
        """
        Cut image
        :param img:
        :return:
        """
        left = img.shape[1]
        top = img.shape[0]
        right, bottom = 0, 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] != 255:
                    left = min(left, j)
                    right = max(right, j)
                    top = min(top, i)
                    bottom = max(bottom, i)
        length = bottom - top
        width = right - left
        left = max(0, left - int(length / 2))
        right = min(right + int(length / 2), img.shape[1])
        top = max(0, top - int(width / 2))
        bottom = min(bottom + int(width / 2), img.shape[0])
        print(str(left) + "," + str(right) + "," + str(top) + "," + str(bottom))
        img = img[top:bottom, left:right]
        return img

    def get_best_shift(self, img):
        cy, cx = ndimage.measurements.center_of_mass(img)
        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)
        return shiftx, shifty

    def shift(self, img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted

    def rotate(self, image):
        """
        Rotate image and flip it
        :param image:
        :return:
        """
        image = image.reshape([HEIGHT, WIDTH])
        image = np.fliplr(image)
        image = np.rot90(image)
        return image

    def split_letters(self, img):
        """
        Method that takes an images and splits all the letters in the image, it does this by using contours
        :param image:
        :return:
        """
        images = []
        image = cv2.imread(img)
        height, width, depth = image.shape

        # resizing the image to find spaces better
        image = cv2.resize(image, dsize=(width * 5, height * 4), interpolation=cv2.INTER_CUBIC)

        # grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # binary
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # dilation
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)

        # adding GaussianBlur
        gsblur = cv2.GaussianBlur(img_dilation, (5, 5), 0)

        # find contours
        ctrs, hier = cv2.findContours(gsblur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        m = list()
        # sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        pchl = list()
        dp = image.copy()

        for i, ctr in enumerate(sorted_ctrs):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(dp, (x - 10, y - 10), (x + w + 10, y + h + 10), (90, 0, 255), 9)
            crop_img = image[y:y + h, x:x + w]
            bordersize = 700
            border = cv2.copyMakeBorder(
                crop_img,
                top=bordersize,
                bottom=bordersize,
                left=bordersize,
                right=bordersize,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )
            images.append(border)
            # border = cv2.resize(border, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("cropped{0}.jpg".format(i), border)

            # roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            # roi = np.array(roi)
            plt.imshow(border)
            plt.show()
        cv2.imwrite("boxes.jpg", dp)
        plt.imshow(dp)
        plt.show()
        return images
