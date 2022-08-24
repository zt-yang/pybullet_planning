import numpy as np
import time
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class FeasibilityChecker(object):

    def __init__(self, **kwargs):
        raise NotImplementedError('should implement this for FeasibilityChecker')

    def __call__(self, *args, **kwargs):
        return self.check(*args, **kwargs)

    def check(self, input):
        raise NotImplementedError('should implement this for FeasibilityChecker')


class Oracle(FeasibilityChecker):

    def __init__(self, label):
        self.answer = not label

    def check(self):
        return self.answer


class Random(FeasibilityChecker):

    def __init__(self):
        np.random.seed(time.time())

    def check(self):
        return np.random.rand() > 0.5


class ModelClassifier(FeasibilityChecker):

    def __init__(self, model):
        self.model = model

    def check(self, input):
        input = input.to(device)
        return bool(self.model(input).detach().numpy())


