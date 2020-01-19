from base.base_model import BaseModel
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from constants import HEIGHT, WIDTH


class XyEMnistModel(BaseModel):
    def __init__(self, config):
        super(XyEMnistModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        # Import the sequential model from keras to build a neural network
        self.model = Sequential()

        # Here we are going to add a convolutional layer with 128 filters with a size of 5x5, an activation function
        # relu, and input shape from a dimension 28x28. So this means we have an input layer, and a convolutional layer
        # at this point. Two in total :).
        # size: n*28*28*1
        self.model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same',
                              activation='relu', input_shape=(HEIGHT, WIDTH, 1)))
        # Then we add a pooling layer, A pooling layer is a new layer added after the convolutional layer.
        # Specifically, after a nonlinearity (e.g. ReLU) has been applied to the feature maps output by a convolutional layer
        # size: n*28*28*128
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Here we add another convolutional with more filters with activation function
        # size: n*14*14*128
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # size: n*14*14*64
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # size: n*7*7*64

        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # size: n*7*7*64
        self.model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
        # size: n*7*7*64

        # Flatten data to 1 dimension
        self.model.add(Flatten())

        # Add a "hidden layer with 128 neurons and a RELU activation function"
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(.5))

        # Finally we have our output layer with 47 neurons for our 47 classes,
        # for that we use the softmax activation function
        self.model.add(Dense(47, activation='softmax'))

        self.model.compile(
            loss=self.config.model.loss,
            optimizer=self.config.model.optimizer,
            metrics=['accuracy'])
