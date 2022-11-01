import gym
import time
import numpy as np
from amalearn.environment import EnvironmentBase

class StockMarketEnvironment(EnvironmentBase):
    def __init__(self,P,P_Lowest,P_Highest,id,discount_factor=1,initial_state=None,container=None):

        state_space = gym.spaces.MultiDiscrete([11,11,11,21])
        action_space = gym.spaces.MultiDiscrete([1,1,1])

        super(StockMarketEnvironment, self).__init__(action_space, state_space, id, container)
        self.gamma = discount_factor
        self.P = P
        self.P_Lowest = P_Lowest
        self.P_Highest = P_Highest
        if initial_state is not None :
            self.obs = initial_state 
        else : 
            self.obs = state_space.sample()
        self.info = {'status': None}

    def calculate_prob(self,state,new_state):
        p = 1
        for i in range(len(state)-1):
            if state[i] == 10 :
                p1 = int(new_state[i]<state[i])*self.P_Highest[1] + int(new_state[i]==state[i])*self.P_Highest[2]
            elif state[i] == 1:
                p1 = int(new_state[i]>state[i])*self.P_Lowest[0] + int(new_state[i]==state[i])*self.P_Lowest[2]
            else:
                p1 = int(new_state[i]>state[i])*self.P[i][0] + int(new_state[i]<state[i])*self.P[i][1] + int(new_state[i]==state[i])*self.P[i][2]
            p = p * p1
        return p 

    def get_dynamics(self,state, action):
        state = np.array(state)
        action = np.array(action)
        adjacent_states = []
        rewards = []
        probabilities = []
        for i in range(-1,2):
            if 0 < state[0] + i < 11:
                for j in range(-1,2):
                    if 0 < state[1] + j < 11:
                        for k in range(-1,2):
                            if 0 < state[2] + k < 11 :
                                if 1 < state[3] < 20 : 
                                    new_state = np.array([state[0]+i,state[1]+j,state[2]+k,state[3]])
                                    rew = 0
                                    if state[3] >= sum(action*state[0:3]):  rew = sum(action * (new_state[0:3] - state[0:3]))
                                    new_state[3] += rew
                                    adjacent_states.append(tuple(new_state))
                                    rewards.append(rew + int(new_state[3] == 1)*(-100) + int(new_state[3] == 20)*100)
                                    probabilities.append(self.calculate_prob(state,new_state))
        return adjacent_states, rewards, probabilities

    def update_values(self,adjacent_states,rewards,probabilities):
        sum = 0
        for i,s_next in enumerate(adjacent_states):
            sum += probabilities[i] * (rewards[i] + self.gamma * self.V[s_next])
        return sum
    
    def Policy_Iteration(self,V,pi,theta):
        self.V = V 
        self.pi = pi 
        all_states = self.get_all_states()
        all_actions = self.get_available_actions()
        Theta = theta
        counter = 1
        policy_stable = False
        while (not policy_stable):
            #Iterative policy Evaluation 
            print("Number of Runs:",counter)
            counter += 1
            Delta = theta + 1 
            
            while( Delta >= Theta) : 
                Delta = 0 
                for s in all_states :
                    old_value = self.V[s]
                    action = self.pi[s]
                    adj_states, rews, probs  = self.get_dynamics(s,action)
                    new_value = self.update_values(adj_states,rews,probs)
                    Delta = max(Delta, abs(old_value - new_value))
                    self.V[s] = new_value
                print("The last Delta in this run is :",Delta)

            #Policy Improvement 
            cnt = 0 
            policy_stable = True
            
            for s in all_states: 
                old_action = self.pi[s]
                state_actions_value = []
                for a in all_actions :
                    adj_states, rew, probs  = self.get_dynamics(s,a)
                    new_value = self.update_values(adj_states,rew,probs)
                    state_actions_value.append(new_value)
                new_action = all_actions[np.argmax(state_actions_value)]
                if old_action != new_action  : 
                    policy_stable = False 
                    cnt += 1 
                self.pi[s] = new_action
            print("Number of total changes in policy is :",cnt)
        return self.V, self.pi 
    
    def calculate_reward(self, action):
        return 

    def terminated(self):
        done = False
        if self.state[3] == 20 :
            self.info['status'] = "Congrats! You achieved the goal."
            done = True
        elif self.state[3] == 1:
            self.info['status'] = "Sorry! You are out of cash."
            done = True
        return done 

    def get_all_states(self):
        all_states = [(i+1,j+1,k+1,l+1) for i in range(10) for j in range(10) for k in range(10) for l in range(20)]
        return all_states
    
    def get_current_state(self):
        return self.obs
    
    def observe(self):
        return self.obs

    def get_available_actions(self):
        available_actions = [(i,j,k) for i in range(2) for j in range(2) for k in range(2)]
        available_actions.remove((1,1,1))
        return available_actions

    def get_states_size(self):
        return self.observation_space.nvec[0]*self.observation_space.nvec[1]*self.observation_space.nvec[2]*self.observation_space.nvec[3]

    def next_state(self, action):
        pass 

    def reset(self):
        pass 

    def render(self, mode='human'):
        pass 

    def close(self):
        return
