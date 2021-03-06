from base.base_evaluator import BaseEvaluator
import numpy as np
import cv2
from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from data_visualizer.simple_mnist_data_visualizer import SimpleMnistDataVisualizer
from pre_processor.image_preprocessor import ImagePreProcessor
from utils.plotter import Plotter
import sklearn.metrics as metrics


class ConvMnistDataPredictor(BaseEvaluator):

    def __init__(self, model, data, map, config, weights=''):
        super(ConvMnistDataPredictor, self).__init__(model, data, config)
        self.map = map
        self.plotter = Plotter()
        if config.evaluator.custom_weight and len(weights) > 0: # Load a custom weight if there is any
            self.model.load_weights(weights)

    def evaluate_model(self):
        """
        Get loss and accuracy of the present model
        """
        X_test, y_test = self.data
        score = self.model.evaluate(X_test, y_test, verbose=1)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def confusion_matrix(self):
        """
        Get confusion matrix
        """
        X_test, y_test = self.data
        y_pred = self.model.predict(X_test)
        y_pred = (y_pred > 0.5)
        cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        print("cm")
        print(cm)

    def predict(self, image):
        """
        Predict a character based on an image
        :param image:
        """
        img = cv2.imread(image) # Read image
        img = cv2.resize(img, (28, 28)) # Resize
        arr = np.array(img).reshape((28, 28, 3)) # Reshape
        arr = np.expand_dims(arr, axis=0)
        prediction = self.model.predict(arr)[0] # Get prediction
        bestclass = ''
        bestconf = -1
        for n in range(47):
            if (prediction[n] > bestconf):
                bestclass = str(n)
                bestconf = prediction[n]
        print('I think this digit is a ' + chr(self.map[int(bestclass)]) + ' with ' + str(
            bestconf * 100) + '% confidence.')

    def predict_from_data_set(self):
        """
        Predict a character based on an element of the test set
        """
        print("Predicting")
        print(chr(self.map[np.argmax(self.data[1][0])]))
        test = self.data[0]
        prediction = self.model.predict(test[:4])[0]
        bestclass = ''
        bestconf = -1
        for n in range(47):
            if (prediction[n] > bestconf):
                bestclass = str(n)
                bestconf = prediction[n]
        print('I think this digit is a ' + chr(int(self.map[int(bestclass)])) + ' with ' + str(
            bestconf * 100) + '% confidence.')

    def predict_ocr(self, image, readImage=True):
        """
        Predict a character based on an image
        :param image:
        :param readImage:
        :return:
        """
        image_pre_processor = ImagePreProcessor()
        img = image_pre_processor.execute(image, readImage) # Pre Process image
        if len(img) > 0:
            img = np.resize(img, (28, 28, 1)) # Resize image
            im2arr = np.array(img) # Transform to np array
            im2arr = np.expand_dims(im2arr, axis=0)
            im2arr = im2arr.astype('float32')
            im2arr /= 255 # Normalize
            prediction = self.model.predict(im2arr)[0] # Prediction
            print("Prediction")
            print(prediction)
            bestclass = ''
            bestconf = -1
            for n in range(47):
                if (prediction[n] > bestconf):
                    bestclass = str(n)
                    bestconf = prediction[n]

            print('I think this digit is a ' + chr(self.map[int(bestclass)]) + ' with ' + str(
                bestconf * 100) + '% confidence.')
            return chr(self.map[int(bestclass)])
        else:
            return ""

    def ocr(self, image):
        """
        Predict each character from an image, it first splits the characters of the image and then makes the prediction for each image
        :param image:
        :return:
        """
        predicted_values = [] # Array to store all the predicted characters
        image_pre_processor = ImagePreProcessor()
        images = image_pre_processor.split_letters(image) # Get the different characters from an image
        for image in images:
            predicted = self.predict_ocr(image, False) # Get prediction
            if len(predicted) > 0:
                predicted_values.append(predicted)
        return predicted_values
