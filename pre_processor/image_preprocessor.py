import math

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

from constants import HEIGHT, WIDTH
from utils.plotter import Plotter


class ImagePreProcessor():

    def __init__(self):
        self.plotter = Plotter()

    def execute(self, image, readImage=True):
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
        self.plotter.plot_image(gray)
        gray = self.cut(gray)
        if len(gray) > 0:
            # resize the images and invert it (black background)
            gray = cv2.resize(255 - gray, (28, 28))
            self.plotter.plot_image(gray)
            cv2.imwrite("test_images/image_1.png", gray)
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
        image = image.reshape([HEIGHT, WIDTH])
        image = np.fliplr(image)
        image = np.rot90(image)
        return image

    def split_letters(self, image):
        images = []
        height = 512
        width = 512
        blank_image = np.zeros((height, width, 1), np.uint8) + 255
        img = cv2.imread(image)
        original = img.copy()
        print("img shape=", img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        cv2.imshow('gray', gray)
        cv2.imshow('thresh', thresh)
        cv2.imwrite("gray.jpg", gray)
        cv2.imwrite("thresh.jpg", thresh)
        cv2.imwrite("white.jpg", blank_image)
        # contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
        cv2.imwrite("contours.jpg", img)

        bb_list = []
        for c in contours:
            bb = cv2.boundingRect(c)
            # save all boxes except the one that has the exact dimensions of the image (x, y, width, height)
            if bb[0] == 0 and bb[1] == 0 and bb[2] == img.shape[1] and bb[3] == img.shape[0]:
                continue
            bb_list.append(bb)

        img_boxes = img.copy()
        i = 0
        bb_list.sort(key=lambda x: x[0])
        for bb in bb_list:
            x, y, w, h = bb
            cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
            crop_img = original[y:y + h, x:x + w]
            bordersize = 100
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
            cv2.imwrite("cropped{0}.jpg".format(i), border)
            i += 1
        cv2.imwrite("boxes.jpg", img_boxes)
        return images
