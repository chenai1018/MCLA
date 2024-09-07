class Data(object):
    def __init__(self, conf, training, test):
        self.config = conf
        self.training_data = training
        self.test_data = test


class Rating(object):
    def __init__(self, conf, data):
        self.config = conf
        self.data = data




