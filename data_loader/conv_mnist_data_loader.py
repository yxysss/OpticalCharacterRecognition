from tensorflow.keras.utils import to_categorical

from base.base_data_loader import BaseDataLoader
from tensorflow.keras.datasets import mnist


class ConvMnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ConvMnistDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.X_train = self.X_train.reshape((-1, 28, 28, 1))
        self.X_test = self.X_test.reshape((-1, 28, 28, 1))
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_train_data_original(self):
        return self.X_train, self.y_train
