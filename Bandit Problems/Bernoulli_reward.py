from amalearn.reward import RewardBase
import numpy as np

class BernoulliReward(RewardBase):
    def __init__(self, p, reward):
        super(BernoulliReward, self).__init__()
        self.p = p
        self.reward = reward
    
    def get_reward(self):
        return self.reward * int(np.random.rand() < self.p) 
