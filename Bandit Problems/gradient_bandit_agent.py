import numpy as np
from amalearn.agent import AgentBase

class GradientBanditAgent(AgentBase):
    def __init__(self, id, environment, learning_rate = 0.1,learning_rate_decay = 1,alpha = 1,beta = 1,gamma = 1,optimistic = False):
        super(GradientBanditAgent, self).__init__(id, environment)
        available_actions = self.environment.available_actions()
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.avg_rew = 0   
        self.N = 0      
        self.H = np.zeros((available_actions))
        self.P = (1/available_actions)*np.ones((available_actions))

    def update(self,action,utility):
        self.learning_rate = self.learning_rate * self.learning_rate_decay
        self.N += 1
        self.avg_rew += (utility - self.avg_rew)/self.N

        for i in range(len(self.H)):
            if i == action :
                self.H[i] += self.learning_rate*(utility - self.avg_rew)*(1-self.P[i])

            else:
                self.H[i] -=  self.learning_rate*(utility - self.avg_rew)*(self.P[i])
        for i in range(len(self.P)):
            self.P[i] = np.exp(self.H[i]) / np.sum(np.exp(self.H))
        

    def get_all(self):
        return self.H,self.P

    def utility_function(self,reward):
        if reward >= 0 :
            u = reward ** self.alpha
        else:
            u =  -1 * self.gamma * ((-reward)** self.beta)
        return u 


    def take_action(self) -> (object, float, bool, object):
        available_actions = self.environment.available_actions()
        action = int(np.random.choice(list(range(available_actions)), size=1,p=self.P))        
        obs, r, d, i = self.environment.step(action)
        print(obs, r, d, i)
        u = self.utility_function(r)
        self.update(action,u)
        self.environment.render()
        return r, action , u
