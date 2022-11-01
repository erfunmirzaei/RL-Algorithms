import numpy as np
import  random
from amalearn.agent import AgentBase

class Two_step_Exp_SARSA_Agent(AgentBase):
    def __init__(self, id, environment,states,actions,epsilon,learning_rate,gamma,dec_eps = None,dec_alph = None):
        super(Two_step_Exp_SARSA_Agent, self).__init__(id, environment)
        self.epsilon = epsilon
        self.t = 0
        self.dec_eps = dec_eps

        self.states = states
        self.actions = actions
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        
        self.Q = {(s,a):random.random() for s in self.states for a in self.actions}
        self.s = random.choice(self.states)

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.dec_alph = dec_alph

    def set_current_state(self,state):
        self.s = state

    def get_current_state(self):
        return self.s

    def take_action(self,action):
        obs, r, d, i = self.environment.step(action)
        #print(obs, r, d, i)
        self.set_current_state(obs)
        #self.environment.render()
        return obs, r, d, i
    
    def choose_action(self):
        probs = []
        Q = [self.Q[(self.s,a)] for a in self.actions]
        a_star = np.argmax(Q)
        for a in self.actions:
            if a == a_star:
                probs.append(1 - self.epsilon + (self.epsilon/ self.num_actions))
            else:
                probs.append(self.epsilon / self.num_actions)

        action = random.choices(self.actions,k=1,weights=probs)
        return action[0]

    def decay_epsilon(self): 
        self.epsilon = self.dec_eps[0]/(self.dec_eps[1] + self.dec_eps[2]*self.t)
        self.t += 1
    
    def decay_learning_rate(self):
        self.learning_rate = self.dec_alph[0]/(self.dec_alph[1] + self.dec_alph[2]*self.t)

    def update(self,S0,A0,R1,S2,R2): 
        Q = [self.Q[(S2,a)] for a in self.actions]
        a_star = np.argmax(Q)
        Exp_Q = 0
        for a in self.actions:
            if a == a_star:
                Exp_Q += (1 - self.epsilon + (self.epsilon/ self.num_actions)) * self.Q[(S2,a)]
            else:
                Exp_Q += (self.epsilon / self.num_actions)* self.Q[(S2,a)]

        self.Q[(S0,A0)] += self.learning_rate*(R1 + self.gamma * R2 + (self.gamma **2) * Exp_Q  - self.Q[(S0,A0)])
