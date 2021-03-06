class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None

    def save(self, checkpoint_path):
        """
        save function that saves the checkpoint in the path defined in the config file
        :param checkpoint_path:
        """
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    def load(self, checkpoint_path):
        """
        load latest checkpoint from the experiment path defined in the config file
        :param checkpoint_path:
        """
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        raise NotImplementedError
