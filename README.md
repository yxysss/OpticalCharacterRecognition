# Optical-Character-Recognition
Optical Character Recognition System, employing Convolutional Neural Network models

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
