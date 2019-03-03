import numpy as np
import random

class OU_Noise(object):
    '''produces a noise that has an inertia'''

    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.3):
        self.dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim)*mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.uniform(low=-1, high=1, size=(len(x)))
        self.state = x + dx
        return self.state

    def reset(self):
        self.state = np.ones(self.dim)*self.mu