import gym
from amalearn.environment import EnvironmentBase

class NetworkEnvironment(EnvironmentBase):
    def __init__(self, rewards, id, container=None):
        state_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(len(rewards))

        super(NetworkEnvironment, self).__init__(action_space, state_space, id, container)
        self.arms_rewards = rewards
        self.state = {
            'length': 0,
            'last_action': None
        }

    def calculate_reward(self, action):
        a_0 = self.arms_rewards[action][0].get_reward() 
        a_1 = self.arms_rewards[action][1].get_reward() 
        a_2 = self.arms_rewards[action][2].get_reward() 
        a_3 = self.arms_rewards[action][3].get_reward() 
        a_4 = self.arms_rewards[action][4].get_reward() 
        a_5 = self.arms_rewards[action][5].get_reward() 
        a_6 = self.arms_rewards[action][6].get_reward() 
        return -a_0 - a_1 - a_2 - a_3 - a_4 - a_5 - a_6

    def terminated(self):
        return True

    def observe(self):
        return {}

    def available_actions(self):
        return self.action_space.n

    def next_state(self, action):
        self.state['length'] += 1
        self.state['last_action'] = action

    def reset(self):
        self.state['length'] = 0
        self.state['last_action'] = None

    def render(self, mode='human'):
        print('{}:\taction={}'.format(self.state['length'], self.state['last_action']))

    def close(self):
        return
        

