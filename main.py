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
    global image
    try:
        args = get_args()
        config = process_config(args.config)
        image = process_image(args.image) if args.image is not 'None' else None
    except:
        print("missing or invalid arguments (check correct config or image paths)")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir, config.callbacks.history_dir])

    print('Create the data generator.')
    data_loader = ConvEMnistDataLoader(config)  # Load data set

    print('Some data visualization')
    X_train, y_train = data_loader.get_train_data()  # Get training data
    print("ytrain")
    print(y_train.shape)
    mapp = data_loader.get_map()  # Get map dictionary (Refer to emnist-balanced-mapping.txt file)
    data_visualizer = SimpleMnistDataVisualizer(X_train, y_train, mapp)
    data_visualizer.plot_first_digit()  # Plot first character of training set
    data_visualizer.plot_range()  # Plot several characters

    print('Create the model.')
    model = ConvEMnistModel(config)  # Create the model based on configuration file

    print("Model Summary")
    model.model.summary()  # Print a summary of the model with the respective parameters

    # Custom weight to use instead of training the model
    weight = config.evaluator.weight

    print('Create the trainer')
    trainer = ConvMnistModelTrainer(model.model, data_loader.get_train_data(), config)

    if not config.evaluator.custom_weight:
        print('Start training the model.')
        # if not config.evaluator.custom_weight:
        trainer.train()

        print("Plot loss and accuracy in training model")
        data_visualizer.plot_loss_acc()

        print("Finish training")

    print("Predict")

    predictor = ConvMnistDataPredictor(model.model, data_loader.get_test_data(), mapp, config, weight)
    predict_image = image if image is not None else './test_images/data_representation/0.png'
    predicted_values = predictor.ocr(predict_image)
    print("Predicted values")
    print(predicted_values)
    # predictor.predict3('./test_images/h/1.png')
    # predictor.predict_from_data_set()

    """
    Evaluate model with test set
    """
    predictor.evaluate_model()

    predictor.confusion_matrix()


if __name__ == '__main__':
    main()
    # app.run()
