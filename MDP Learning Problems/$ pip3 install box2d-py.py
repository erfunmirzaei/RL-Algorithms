$ pip3 install box2d-py
$ pip3 install gym[Box_2D]
import gym
env = gym.make("LunarLander-v2")
env.reset()
env.render()

for _ in range(4):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action) # Take a random action
    print(info)
    print('action:',action)
    env.render()
env.close()