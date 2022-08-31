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

    def __init__(self, correct):
        self.correct = correct

    def check(self, input):
        if len(input) != len(self.correct):
            return False
        for i in range(len(input)):
            action = [input[i].name] + list(input[i].args)
            for j in range(len(self.correct[i])):
                if '=' not in self.correct[i][j]:
                    # if len(action) != len(self.correct[i]):
                    #     print('len(input[i]) != len(self.correct[i])', action, self.correct[i])
                    if str(action[j]) != self.correct[i][j]:
                        print(f'\n\nOracle feasibility checker | correct', self.correct[i],
                              'input', action, '\n\n')
                        return False
        return True


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


