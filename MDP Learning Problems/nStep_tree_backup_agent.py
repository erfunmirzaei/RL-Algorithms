import numpy as np
import  random
from amalearn.agent import AgentBase

class nStep_tree_backup_Agent(AgentBase):
    def __init__(self, id, environment,states,actions,n,epsilon,learning_rate,gamma,dec_eps = None,dec_alph = None):
        super(nStep_tree_backup_Agent, self).__init__(id, environment)
        self.n = n
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

    def calculate_G_Expected(self,R,S):
        Q = [self.Q[(S,a)] for a in self.actions]
        a_star = np.argmax(Q)
        G = R
        for a in self.actions:
            if a == a_star:
                G += self.gamma * (1 - self.epsilon + (self.epsilon/ self.num_actions)) * self.Q[(S,a)]
            else:
                G += self.gamma * (self.epsilon / self.num_actions)* self.Q[(S,a)]
        return G 

    def calculate_G_treebackup(self,R,S,A,G):
        Q = [self.Q[(S,a)] for a in self.actions]
        a_star = np.argmax(Q)
        G_new = R
        for a in self.actions:
            if a != A :     
                if a == a_star:
                    G_new += self.gamma * (1 - self.epsilon + (self.epsilon/ self.num_actions)) * self.Q[(S,a)]
                else:
                    G_new += self.gamma * (self.epsilon / self.num_actions)* self.Q[(S,a)]
            else: 
                if a == a_star:
                    G_new += self.gamma * (1 - self.epsilon + (self.epsilon/ self.num_actions)) * G
                else:
                    G_new += self.gamma * (self.epsilon / self.num_actions)* G            
        return G_new 

    def update_Q(self,G,s_tau,a_tau): 
        self.Q[(s_tau,a_tau)] += self.learning_rate*(G  - self.Q[(s_tau,a_tau)])
