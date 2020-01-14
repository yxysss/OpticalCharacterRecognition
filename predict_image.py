from data_loader.conv_emnist_data_loader import ConvEMnistDataLoader
from data_loader.conv_mnist_data_loader import ConvMnistDataLoader
from data_loader.simple_mnist_data_loader import SimpleMnistDataLoader
from data_visualizer.simple_mnist_data_visualizer import SimpleMnistDataVisualizer
from evaluater.conv_mnist_data_predictor import ConvMnistDataPredictor
from models.conv_emnist_model import ConvEMnistModel
from models.conv_mnist_model import ConvMnistModel
from models.simple_mnist_model import SimpleMnistModel
from trainers.conv_mnist_trainer import ConvMnistModelTrainer
from trainers.simple_mnist_trainer import SimpleMnistModelTrainer
from utils.config import process_config, process_image
from utils.dirs import create_dirs
from utils.utils import get_args
import numpy as np
import sklearn.metrics as metrics


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    global image, config
    try:
        args = get_args()
        config = process_config("configs/conv_emnist_from_config.json")
        image = process_image(args.image)
    except:
        print("missing or invalid arguments (check image path)")
        exit(0)

    # create the experiments dirs
    data_loader = ConvEMnistDataLoader(config)
    mapp = data_loader.get_map()

    print('Create the model.')
    model = ConvEMnistModel(config)  # Create the model based on configuration file

    # Custom weight to use instead of training the model
    weight = "./experiments/2019-12-15/conv_emnist_from_config/checkpoints/conv_emnist_from_config-10-0.35.hdf5"

    predictor = ConvMnistDataPredictor(model.model, [], mapp, config, weight)
    predict_image = image if image is not None else './test_images/data_representation/0.png'
    predicted_values = predictor.ocr(predict_image)
    print("Predicted values")
    print(predicted_values)


if __name__ == '__main__':
    main()
    # app.run()
