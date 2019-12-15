from pre_processor.image_preprocessor import ImagePreProcessor
import matplotlib.pyplot as plt

if __name__ == '__main__':

    imagePreProcessor = ImagePreProcessor()
    img = imagePreProcessor.execute('./test_images/0/1.png')
    plt.imshow(img)
