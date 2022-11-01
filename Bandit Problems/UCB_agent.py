import numpy as np
from math import sqrt
from amalearn.agent import AgentBase

class UCBAgent(AgentBase):
    def __init__(self, id, environment, exploration_degree = 2,alpha = 1,beta = 1,gamma = 0.1):
        super(UCBAgent, self).__init__(id, environment)
        available_actions = self.environment.available_actions()
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.exp_deg = exploration_degree                        #hyperparameter
        self.Q = np.zeros((available_actions,1))                 #action value fuction(expected reward) for each arm
        self.N = np.zeros((available_actions,1))                 #number of doing each arm
        self.trials = 0                                          #number of total trials 
        self.UCBs = 1000000000 * np.ones((available_actions,1))  #Upper confidence bound for each arm

    def update(self,action,utility):
        self.trials += 1
        self.N[action] += 1
        self.Q[action] += (utility - self.Q[action])/self.N[action] 
        self.UCBs[action] = self.Q[action] + self.exp_deg * sqrt(np.log(self.trials)/self.N[action])
    
    def utility_function(self,reward):
        if reward >= 0 :
            u = reward ** self.alpha
        else:
            u =  -1 * self.gamma * ((-reward)** self.beta)
        return u 
    
    def take_action(self) -> (object, float, bool, object):
        action = np.argmax(self.UCBs)
        obs, r, d, i = self.environment.step(action)       
        print(obs, r, d, i) 
        u = self.utility_function(r)
        self.update(action,u) 
        self.environment.render()
        return r, action, u
