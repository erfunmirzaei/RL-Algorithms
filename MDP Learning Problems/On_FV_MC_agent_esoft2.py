import numpy as np
import  random
from amalearn.agent import AgentBase

class OnPolicy_esoft_FirstVisit_MC_Agent(AgentBase):
    def __init__(self, id, environment,states,actions, epsilon,dec_eps = None):
        super(OnPolicy_esoft_FirstVisit_MC_Agent, self).__init__(id, environment)
        self.epsilon = epsilon
        self.t = 0
        self.dec_eps = dec_eps

        self.states = states
        self.actions = actions
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        
        self.Q = {(s,a):random.random() for s in self.states for a in self.actions}
        self.Returns = {(s,a):[] for s in self.states for a in self.actions}

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
                probs.append(1 - self.epsilon + (self.epsilon/ self.num_actions))
            else:
                probs.append(self.epsilon / self.num_actions)

        action = random.choices(self.actions,k=1,weights=probs)
        #obs, r, d, i = (self.environment.P[self.s][action[0]])[0]
        obs, r, d, i = self.environment.step(action[0])
        #print(obs, r, d, i)
        self.set_current_state(obs)
        #self.environment.render()
        return obs, r, d, i, action[0]
    
    def decay_epsilon(self):
        self.epsilon = self.dec_eps[0]/(self.dec_eps[1] + self.dec_eps[2]*self.t)
        self.t += 1

    def update(self,G,state,action):
        self.Returns[(state,action)].append(G)
        self.Q[(state,action)] = np.average(self.Returns[(state,action)])
