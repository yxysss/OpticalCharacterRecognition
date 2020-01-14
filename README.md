# Optical Character Recognition
Optical Character Recognition System, employing Convolutional Neural Network models. Data Representation, Reduction and Analysis course.

## Install
All requirements can be installed using pip.
```
pip install -r requirements.txt
```
## Run
- A configuration file needs to be passed in order to run the project, this configuration file contains the parameters for building, training and evaluating the model (Refer to configs/conv_emnist_from_config.json)
- An image to be predicted can be passed as an optional parameter (-i <path_to_image>)
#### Run model trainer
The configuration file conv_emnist_from_config_train.json has the "custom_weight" parameter set to false. Therefore the model will be trained
```
"evaluator": {
    "name": "conv_mnist_data_predictor.ConvMnistDataPredictor",
    "weight" : "",
    "custom_weight" : false
  }
```

```
python main.py -c configs/conv_emnist_from_config_train.json -i test_images/hello/hello.png
```
#### Run model predictor
The configuration file conv_emnist_from_config.json has the "custom_weight" parameter set to true. Therefore the model will use weights already trained and predict the image. (Don't forget to add the path of the file containing the trained weights in the configuration file)
```
"evaluator": {
    "name": "conv_mnist_data_predictor.ConvMnistDataPredictor",
    "weight" : "./experiments/2019-12-15/conv_emnist_from_config/checkpoints/conv_emnist_from_config-10-0.35.hdf5",
    "custom_weight" : true
  }
```
```
python main.py -c configs/conv_emnist_from_config.json -i test_images/hello/hello.png
```
To quickly predict an image with our best weights already trained execute the following script with the image path
 ```
 python predict_image.py -i test_images/hello/hello.png
 ```
#### Configuration file
Configuration file example
```
{
  "exp": {
    "name": "conv_emnist_from_config"
  },
  "data_loader": {
    "name": "conv_emnist_data_loader.ConvEMnistDataLoader"
  },
  "model": {
    "name": "conv_emnist_model.ConvEMnistModel",
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "categorical_crossentropy"
  },
  "trainer": {
    "name": "simple_mnist_trainer.SimpleMnistModelTrainer",
    "num_epochs": 10,
    "batch_size": 512,
    "validation_split": 0.25,
    "verbose_training": true
  },
  "evaluator": {
    "name": "conv_mnist_data_predictor.ConvMnistDataPredictor",
    "custom_weight" : true
  },
  "callbacks": {
    "checkpoint_monitor": "val_loss",
    "checkpoint_mode": "min",
    "checkpoint_save_best_only": true,
    "checkpoint_save_weights_only": true,
    "checkpoint_verbose": true,
    "tensorboard_write_graph": true
  }
}
```
## Project Structure
The following structure was based on this template : https://github.com/Ahmkel/Keras-Project-Template
```
├── main.py             - Class responsible for training and testing the model.
│
│
├── base                - This folder contains the abstract classes of the project components
│   ├── base_data_loader.py   - This file contains the abstract class of the data loader.
│   ├── base_model.py   - This file contains the abstract class of the model.
|   ├── base_evaluator.py   - This file contains the abstract class of the evaluator of model.
│   └── base_train.py   - This file contains the abstract class of the trainer.
│
│
├── model               - this folder contains the model of the project (The structure of the neural network and the hyper parameters).
│   └── conv_emnist_model.py
│
│
├── trainer             - this folder contains the trainer of the project.
│   └── conv_mnist_trainer.py
│
|
├── data_loader         - this folder contains the data loader of the project.
│   └── conv_emnist_data_loader.py
│
│
├── configs             - this folder contains the experiment and model configurations of the project.
│   └── simple_mnist_config.json
│
│
├── data_set            - this folder might contain the emnist dataset of the project (Training and Testing).
│
│
└── utils               - this folder contains any utils needed for the project.
     ├── config.py      - util functions for parsing the config files.
     ├── dirs.py        - util functions for creating directories.
     └── utils.py       - util functions for parsing arguments.
```
