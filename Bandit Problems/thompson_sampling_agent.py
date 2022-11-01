import numpy as np
from math import sqrt
from amalearn.agent import AgentBase

class ThompsonSamplingAgent(AgentBase):
    def __init__(self, id, environment):
        super(ThompsonSamplingAgent, self).__init__(id, environment)
        available_actions = self.environment.available_actions()
        self.stds = list(1000000000 * np.ones((available_actions,1)))  #stds of estimated gaussian distributions
        self.means = list( np.zeros((available_actions,1)))            #means of estimated gaussian distributions

    def get_samples(self,means,stds):
        samples = [np.random.normal(means[i],stds[i]) for i in range(len(means))]
        return samples

    def update(self,reward,action):
        new_std = sqrt( 1 / ((1/self.stds[action]**2) + 1) )
        new_mean = (reward + (self.means[action] / self.stds[action] ** 2)) / ((1/self.stds[action]**2) + 1) 
        return new_mean, new_std


    def take_action(self):# -> (object, float, bool, object):
        samples = self.get_samples(self.means,self.stds)
        action = np.argmax(samples)
        obs, r, d, i = self.environment.step(action)
        self.means[action], self.stds[action] = self.update(r,action)
        print(obs, r, d, i)
        self.environment.render()
        return r,action

