import gym
from gym import spaces
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import numpy as np

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C #
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class CustomGymEnv(UnityToGymWrapper, gym.Env):
    '''Create a gym environment corresponding to the UnityToGymWrapper environment'''

    def __init__(self, uEnv:UnityEnvironment):
        self.UnityEnv = super(CustomGymEnv, self).__init__(uEnv)

    def step(self, action):
        state, reward, done, info = super(CustomGymEnv, self).step(action)
        pyreward = reward.item() #transforms a np.float in a naitve float type
        return state, pyreward, done, info
    
    def reset(self):
        observation = super(CustomGymEnv, self).reset()

        return observation
    
    def render(self):
        super(CustomGymEnv, self).render()

    def close(self):
        super(CustomGymEnv, self).close()

def make_env( Genv: CustomGymEnv, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> CustomGymEnv:
        env = Genv
        env.seed(seed + rank)
        return env
    seed = np.random.randint(0,10)
    return _init

# Load the UnityEnvironment
LEVEL = 3
uEnv = UnityEnvironment("/Users/victorbarberteguy/Desktop/3A/INF581/RL/Rider/Build_Level"+str(LEVEL)+".app", worker_id=0, seed=1, no_graphics=True)
env = UnityToGymWrapper(uEnv, uint8_visual = False, flatten_branched= True,  allow_multiple_obs=True)

# Create the custom gym environment

MyEnv = CustomGymEnv(uEnv)

# Test if the environment follows the interface

check_env(MyEnv, warn=True)

#MyEnv = make_vec_env(make_env: MyEnv, n_envs=2)
env = SubprocVecEnv([make_env(MyEnv, i) for i in range(2)])

#env.reset()

# ACCEL = 1
# A = np.ndarray([ACCEL])
# # Hardcoded best agent: always go left!
# n_steps = 200
# for step in range(n_steps):
#   print("Step {}".format(step + 1))
#   obs, reward, done, info = MyEnv.step(A)
#   print('obs=', obs, 'reward=', reward, 'done=', done)
#   if done:
#     print("Goal reached!", "reward=", reward)
#     break

# Train with A2C
model = A2C('MlpPolicy', env, verbose=1).learn(10000)

MyEnv.close()
# def test_model(model, env, n_episodes=30):

#     rewards = []
#     best_reward = 0
#     success = False
#     n_successes = 0
#     for i in range(n_episodes):
#         obs = env.reset()

#         score = 0
#         done = False

#         while not done:
#             action, _ = model.predict(obs)
#             obs, reward, done, info = env.step(action)

#             score += reward
#             print('obs=', obs, 'reward=', reward, 'done=', done)

#             if(reward==200):
#                 success = True

#         if(score < best_reward):
#             best_reward = score
        

#         if(success):
#             n_successes += 1

#         rewards.append(score)

#     reward = np.mean(rewards)
#     print("Training on {} episodes, mean reward = {} and model had {} successes".format(n_episodes, reward, n_successes))
    
#     file = open("ResultsSB3A2C.txt", 'w')
#     settings = ['\nEPISODES  = ' + str(n_episodes),'\nsuccesses  = ' + str(n_successes), '\nMean Reward  = ' + str(reward),'\Best reward  = ' + str(best_reward)]
#     file.writelines(settings)
#     file.close()

#     return None

# test_model(model, MyGymEnv)