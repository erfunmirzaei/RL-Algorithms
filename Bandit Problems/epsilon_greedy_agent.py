import numpy as np
from amalearn.agent import AgentBase

class EpsilonGreedyAgent(AgentBase):
    def __init__(self, id, environment, epsilon = 0.1,epsilon_decay = 1,alpha = 1,beta = 1,gamma = 0.1,optimistic = False):
        super(EpsilonGreedyAgent, self).__init__(id, environment)
        available_actions = self.environment.available_actions()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.Q = np.zeros((available_actions,1))     #action value fuction
        self.N = np.zeros((available_actions,1))     #number of doing an action 

    def get_number_actions(self):
        return self.N

    def update(self,action,utility):
        self.epsilon = self.epsilon * self.epsilon_decay
        self.N[action] += 1
        self.Q[action] += (utility - self.Q[action])/self.N[action] 

    def utility_function(self,reward):
        if reward >= 0 :
            u = reward ** self.alpha
        else:
            u =  -1 * self.gamma * ((-reward)** self.beta)
        return u 
    
    def take_action(self) -> (object, float, bool, object):
        available_actions = self.environment.available_actions()
        rand = np.random.rand()
        if rand < self.epsilon:
            action = np.random.choice(available_actions)
        else:
            action = np.argmax(self.Q)
        
        obs, r, d, i = self.environment.step(action)
        print(obs, r, d, i)
        u = self.utility_function(r)
        self.update(action,u)
        self.environment.render()
        return r, action , u
