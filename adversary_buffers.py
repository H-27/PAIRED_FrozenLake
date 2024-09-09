import numpy as np

class Adversary_Buffer(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []
        self.dones = []
        self.rand_vec = np.zeros(0, dtype=np.float32)

    def storeTransition(self, state, action, reward, value, probs, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.probs.append(probs)
        self.dones.append(done)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []
        self.dones = []
        self.rand_vec = np.zeros(0, dtype=np.float32)

    def calculate_returns(self, gamma):
        g = np.zeros(len(self.rewards))
        for i in range(len(g)):
            g_sum = 0
            discount = 1
            for j in range(i, len(g)):
                g_sum += self.rewards[j] * discount
                discount *= gamma
            g[i] = g_sum
        return g