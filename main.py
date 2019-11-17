from data_loader.simple_mnist_data_loader import SimpleMnistDataLoader
from data_visualizer.simple_mnist_data_visualizer import SimpleMnistDataVisualizer
from models.simple_mnist_model import SimpleMnistModel
from trainers.simple_mnist_trainer import SimpleMnistModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = SimpleMnistDataLoader(config)

    print('Some data visualization')
    X_train_original, y_train = data_loader.get_train_data_original()
    data_visualizer = SimpleMnistDataVisualizer(X_train_original)
    data_visualizer.plot_first_digit()

    print('Create the model.')
    model = SimpleMnistModel(config)

    print("Model Summary")
    model.model.summary()

    print('Create the trainer')
    trainer = SimpleMnistModelTrainer(model.model, data_loader.get_train_data(), config)

    print('Start training the model.')
    trainer.train()

    print("Finish training")
    print("Predict")
    X_test, y_test = data_loader.get_test_data()
    predict = model.model.predict(X_test[:4])
    print(predict)







if __name__ == '__main__':
    main()
