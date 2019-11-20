from keras.utils import to_categorical, np_utils

from base.base_data_loader import BaseDataLoader
from keras.datasets import mnist
import numpy as np
import pandas as pd

from constants import HEIGHT, WIDTH
from pre_processor.image_preprocessor import ImagePreProcessor


class ConvEMnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ConvEMnistDataLoader, self).__init__(config)
        self.image_preprocessor = ImagePreProcessor()
        train = pd.read_csv("./data_set/emnist-balanced-train.csv", delimiter=',')
        test = pd.read_csv("./data_set/emnist-balanced-test.csv", delimiter=',')
        self.mapp = pd.read_csv("./data_set/emnist-balanced-mapping.txt", delimiter=' ', \
                           index_col=0, header=None, squeeze=True)
        print(self.mapp)
        print("Train: %s, Test: %s, Map: %s" % (train.shape, test.shape, self.mapp.shape))
        train_x = train.iloc[:, 1:]
        train_y = train.iloc[:, 0]
        del train
        test_x = test.iloc[:, 1:]
        test_y = test.iloc[:, 0]
        del test
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        train_x = np.asarray(train_x)
        train_x = np.apply_along_axis(self.image_preprocessor.rotate, 1, train_x)
        print("X_train:", train_x.shape)

        test_x = np.asarray(test_x)
        test_x = np.apply_along_axis(self.image_preprocessor.rotate, 1, test_x)
        print("X_test:", test_x.shape)

        train_x = train_x.astype('float32')
        self.X_train = train_x / 255
        test_x = test_x.astype('float32')
        self.X_test = test_x / 255
        num_classes = train_y.nunique()
        print("NUMBER OF CLASSES")
        print(num_classes)
        self.y_train = np_utils.to_categorical(train_y, num_classes)
        self.y_test = np_utils.to_categorical(test_y, num_classes)
        self.X_train = self.X_train.reshape((-1, WIDTH, HEIGHT, 1))
        self.X_test = self.X_test.reshape((-1, WIDTH, HEIGHT, 1))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_map(self):
        return self.mapp

