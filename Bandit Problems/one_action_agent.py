import numpy as np
from amalearn.agent import AgentBase

class OneActionAgent(AgentBase):
    def __init__(self, id, environment,action,alpha = 1,beta = 1,gamma = 0.1):
        super(OneActionAgent, self).__init__(id, environment)
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.action = action
        self.Q = 0
        self.N = 0
    
    def update(self,utility):
        self.N += 1
        self.Q += (utility - self.Q)/self.N 

    def utility_function(self,reward):
        if reward >= 0 :
            u = reward ** self.alpha
        else:
            u =  -1 * self.gamma * ((-reward)** self.beta)
        return u 
        
    def take_action(self) -> (object, float, bool, object):
        obs, r, d, i = self.environment.step(self.action)
        print(obs, r, d, i)
        u = self.utility_function(r)
        self.update(u)
        self.environment.render()
        return r, self.Q ,u
