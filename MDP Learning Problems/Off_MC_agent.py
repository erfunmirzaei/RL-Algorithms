import numpy as np
import  random
from amalearn.agent import AgentBase

class OffPolicy_MC_Agent(AgentBase):
    def __init__(self, id, environment,states,actions, epsilon,dec_eps = None):
        super(OffPolicy_MC_Agent, self).__init__(id, environment)
        self.epsilon = epsilon
        self.t = 0
        self.dec_eps = dec_eps

        self.states = states
        self.actions = actions
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        
        self.Q = {(s,a):random.random() for s in self.states for a in self.actions}
        self.C = {(s,a):0 for s in self.states for a in self.actions}
        
        policy1 = {(s,a):self.epsilon/self.num_actions for s in self.states for a in self.actions}
        for s in self.states:
            Q = [self.Q[(s,a)] for a in self.actions]
            a_star = np.argmax(Q)
            policy1[(s,a_star)] += 1 - self.epsilon
        self.b_policy =  policy1
        
        policy2 = {(s,a):0 for s in self.states for a in self.actions}
        for s in self.states:
            Q = [self.Q[(s,a)] for a in self.actions]
            a_star = np.argmax(Q)
            policy2[(s,a_star)] = 1
        self.p_policy = policy2
        
        self.s = random.choice(self.states)

    def set_current_state(self,state):
        self.s = state

    def get_current_state(self):
        return self.s

    def take_action(self) -> (object, float, bool, object):
        probs = []
        Q = [self.Q[(self.s,a)] for a in self.actions]
        a_star = np.argmax(Q)
        for a in self.actions:
            if a == a_star:
                self.b_policy[(self.s,a)] = 1 - self.epsilon + (self.epsilon/ self.num_actions)
                probs.append(1 - self.epsilon + (self.epsilon/ self.num_actions))
            else:
                self.b_policy[(self.s,a)] = self.epsilon / self.num_actions
                probs.append(self.epsilon / self.num_actions)

        action = random.choices(self.actions,k=1,weights=probs)
        obs, r, d, i = self.environment.step(action[0])
        #print(obs, r, d, i)
        self.set_current_state(obs)
        return obs, r, d, i, action[0]
    
    def decay_epsilon(self):
        self.epsilon = self.dec_eps[0]/(self.dec_eps[1] + self.dec_eps[2]*self.t)
        self.t += 1
    
    def choose_action(self) -> (object, float, bool, object):
        probs = []
        for a in self.actions:
            probs.append(self.p_policy[self.s,a])
        
        action = np.argmax(probs)
        obs, r, d, i = self.environment.step(action)
        self.set_current_state(obs)
        return obs, r, d, i, action

    def update(self,G,W,state,action):
        self.C[(state,action)] += W
        self.Q[(state,action)] += (W/self.C[(state,action)])*(G - self.Q[(state,action)])
        Q = [self.Q[(state,a)] for a in self.actions]
        a_star = np.argmax(Q)
        for a in self.actions:
            if a == a_star:
                self.p_policy[(state,a)] = 1
            else:
                self.p_policy[(state,a)] = 0

        return a_star,self.b_policy[(state,action)]