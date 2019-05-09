import numpy as np
import random

from sklearn import preprocessing
import features

class KerasDataGenerator(object):

    def __init__(self, X, y, n_steps=150, batch_size=16):
        self.X = X
        self.y = y
        self.n_steps = n_steps
        self.n_features = X.shape[1]
        self.batch_size = batch_size

    # generates batch of training data
    def generator(self):
        while True: 
            samples = np.zeros((self.batch_size, self.n_steps, self.n_features))
            targets = np.zeros(self.batch_size,)
            
            indices = random.sample(range(self.n_steps, len(self.data)), self.batch_size)
            for j, ind in enumerate(indices):
                
                # samples[j] = self.create_X(ind)
                samples[j] = self.X[ind - 150:ind]
                targets[j] = self.y[ind - 1]
            yield samples, targets
