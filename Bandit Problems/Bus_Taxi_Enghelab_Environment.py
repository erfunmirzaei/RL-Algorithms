import gym
import numpy as np 
from amalearn.environment import EnvironmentBase

class EnghelabEnvironment(EnvironmentBase):
    def __init__(self, rewards,id, container=None):
        state_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(len(rewards))
        
        super(EnghelabEnvironment, self).__init__(action_space, state_space, id, container)
        self.arms_rewards = rewards
        self.bus_time = np.random.normal(loc = 8, scale= 3)
        self.state = {
            'length': 0,
            'last_action': None
        }

    def calculate_reward(self, action):
        if self.bus_time < action:
            return self.arms_rewards[action][0]
        else :
            return self.arms_rewards[action][1]

    def terminated(self):
        return True 
        
    def observe(self):
        return {}
    
    def get_info(self, action):
        if  self.bus_time < action :
            return {"Bus arrived"}
        else:
            return {"Bus didn't arrive I went by taxi"}
        

    def available_actions(self):
        return self.action_space.n

    def next_state(self, action):
        self.state['length'] += 1
        self.state['last_action'] = action

    def reset(self):
        self.state['length'] = 0
        self.state['last_action'] = None
        self.bus_time = np.random.normal(loc = 8, scale= 3)

    def render(self, mode='human'):
        print('{}:\taction={}'.format(self.state['length'], self.state['last_action']))

    def close(self):
        return
        
