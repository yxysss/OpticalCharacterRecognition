from base.base_evaluator import BaseEvaluator
import numpy as np
import cv2
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

from data_visualizer.simple_mnist_data_visualizer import SimpleMnistDataVisualizer
from pre_processor.image_preprocessor import ImagePreProcessor
from utils.plotter import Plotter


class ConvMnistDataPredictor(BaseEvaluator):

    def __init__(self, model, data, weights=''):
        super(ConvMnistDataPredictor, self).__init__(model, data)
        self.plotter = Plotter()
        if len(weights) > 0:
            self.model.load_weights(weights)

    def predict(self, image):
        img = cv2.imread(image)
        img = cv2.resize(img, (28, 28))
        arr = np.array(img).reshape((28, 28, 3))
        arr = np.expand_dims(arr, axis=0)
        prediction = self.model.predict(arr)[0]
        bestclass = ''
        bestconf = -1
        for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if (prediction[n] > bestconf):
                bestclass = str(n)
                bestconf = prediction[n]
        print('I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.')

    def predict_from_data_set(self):
        print("Predicting")
        print(np.argmax(self.data[1][0]))
        test = self.data[0]
        prediction = self.model.predict(test[:4])[0]
        bestclass = ''
        bestconf = -1
        for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if (prediction[n] > bestconf):
                bestclass = str(n)
                bestconf = prediction[n]
        print('I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.')

    def predict2(self, image):
        image_pre_processor = ImagePreProcessor()
        img = image_pre_processor.execute(image)
        # img = Image.open(image).convert("L")
        # self.plotter.plot_image(img)
        # img = np.resize(img, (28, 28, 1))
        # im2arr = np.array(img)
        # im2arr = np.expand_dims(im2arr, axis=0)
        # im2arr = im2arr.astype('float32')
        # im2arr /= 255
        # prediction = self.model.predict(im2arr)[0]
        # bestclass = ''
        # bestconf = -1
        # for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        #     if (prediction[n] > bestconf):
        #         bestclass = str(n)
        #         bestconf = prediction[n]
        # print('I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.')

    def predict3(self, image):
        image_pre_processor = ImagePreProcessor()
        img = image_pre_processor.execute(image)
        img = np.resize(img, (28, 28, 1))
        im2arr = np.array(img)
        im2arr = np.expand_dims(im2arr, axis=0)
        im2arr = im2arr.astype('float32')
        im2arr /= 255
        prediction = self.model.predict(im2arr)[0]
        print("Prediction")
        print(prediction)
        bestclass = ''
        bestconf = -1
        for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if (prediction[n] > bestconf):
                bestclass = str(n)
                bestconf = prediction[n]
        print('I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.')
