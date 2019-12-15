class BaseEvaluator(object):
    def __init__(self, model, data, config):
        self.config = config
        self.model = model
        self.data = data

    def predict(self, image):
        raise NotImplementedError
