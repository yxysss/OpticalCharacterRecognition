from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import utils as np_utils

from base.base_data_loader import BaseDataLoader
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd

from constants import HEIGHT, WIDTH
from pre_processor.image_preprocessor import ImagePreProcessor


class ConvEMnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ConvEMnistDataLoader, self).__init__(config)
        self.image_preprocessor = ImagePreProcessor()  # Load image pre processor
        train = pd.read_csv("./data_set/emnist-balanced-train.csv", delimiter=',')  # Training data set
        test = pd.read_csv("./data_set/emnist-balanced-test.csv", delimiter=',')  # Testing data set
        self.mapp = pd.read_csv("./data_set/emnist-balanced-mapping.txt", delimiter=' ', \
                                index_col=0, header=None, squeeze=True)  # Map of numbers to letters
        print(self.mapp)
        print("Train: %s, Test: %s, Map: %s" % (train.shape, test.shape, self.mapp.shape))
        train_x = train.iloc[:, 1:]  # Get features of training set
        train_y = train.iloc[:, 0]  # Get label of testing set
        del train
        test_x = test.iloc[:, 1:]  # Get features of testing set
        test_y = test.iloc[:, 0]  # Get label of testing set
        del test
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        train_x = np.asarray(train_x)
        train_x = np.apply_along_axis(self.image_preprocessor.rotate, 1,
                                      train_x)  # Rotate images in training set (Because of how images are presented in original data set)
        print("X_train:", train_x.shape)

        test_x = np.asarray(test_x)
        test_x = np.apply_along_axis(self.image_preprocessor.rotate, 1, test_x)
        print("X_test:", test_x.shape)

        train_x = train_x.astype('float32')  # Cast images to type float
        self.X_train = train_x / 255  # Normalize
        test_x = test_x.astype('float32')
        self.X_test = test_x / 255  # Normalize
        num_classes = train_y.nunique()  # Get number of classes
        print("NUMBER OF CLASSES")
        print(num_classes)
        self.y_train = np_utils.to_categorical(train_y, num_classes)  # Transform to one hot encoding
        self.y_test = np_utils.to_categorical(test_y, num_classes)  # Transform to one hot encoding
        self.X_train = self.X_train.reshape((-1, WIDTH, HEIGHT, 1))  # Reshape training set for conv neural net
        self.X_test = self.X_test.reshape((-1, WIDTH, HEIGHT, 1))  # Reshape test set for conv neural net

    def get_train_data(self):
        """
        Get training data
        :return:
        """
        return self.X_train, self.y_train

    def get_test_data(self):
        """
        Get test data
        :return:
        """
        return self.X_test, self.y_test

    def get_map(self):
        """
        Get map
        :return:
        """
        return self.mapp
