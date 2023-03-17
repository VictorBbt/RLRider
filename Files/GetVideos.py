from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np


uEnv = UnityEnvironment("/Users/victorbarberteguy/Desktop/3A/INF581/RL/Rider/Build_Level1NoMetal.app", worker_id=0, seed=1, no_graphics=False)
env = UnityToGymWrapper(uEnv, uint8_visual = False, flatten_branched= True,  allow_multiple_obs=True)
env.reset()

## play epiosde
EP_DURATION = 100
RENDER = 25

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logistic_regression(s, theta):
    prob_accel = sigmoid(np.dot(s, np.transpose(theta)))
    return prob_accel

def draw_action(s, theta):
    prob_accel = logistic_regression(s, theta)
    r = np.random.random()
    if r < prob_accel:
        return 1
    else:
        return 0
    
def play_one_episode(env, theta, max_episode_length=EP_DURATION):
    s_t = env.reset()

    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_states.append(s_t)

    for t in range(max_episode_length):

        a_t = draw_action(s_t, theta)
        s_t, r_t, done, info = env.step(a_t)

        episode_states.append(s_t)
        episode_actions.append(a_t)
        episode_rewards.append(r_t)

        if done:
            break

    return episode_states, episode_actions, episode_rewards, done


## DEMO

def demo(env, filename):

    
    theta_array = np.load(filename)
    
    print("one theta is stored after one step of gradient descent")

    for i in range(len(theta_array)):
        print("episode for theta at trial number " + str(i) + "i.e the " + str(i*RENDER) + "try.")
        play_one_episode(env, theta_array[i], max_episode_length=EP_DURATION)

    return "demo done"

print(demo(env, "thetas.npy"))

env.close()