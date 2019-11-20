class BaseEvaluator(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def predict(self, image):
        raise NotImplementedError
