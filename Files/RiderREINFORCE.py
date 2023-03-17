from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

# Wrap the environment
uEnv = UnityEnvironment("/Users/victorbarberteguy/Desktop/3A/INF581/RL/Rider/Build_Level1.app", worker_id=0, seed=1, no_graphics=False)
env = UnityToGymWrapper(uEnv, uint8_visual = False, flatten_branched= True,  allow_multiple_obs=True)

##Setting up the rendering environment (where the no_graphiocs = False)
#uEnvR = UnityEnvironment("/Users/victorbarberteguy/Desktop/3A/INF581/RL/Rider/Build.app", worker_id=1, seed=1, no_graphics=False)
#RenderEnv = UnityToGymWrapper(uEnv, uint8_visual = False, flatten_branched= True,  allow_multiple_obs=True)


env.reset()
#RenderEnv.reset()
################ constant policy ##################
# observation = env.reset()
# done = False
# x = []
# y = []

# for t in range(50):


#     action = 1
#     observation, reward, done, truncated = env.step(action)
#     print("tour " + str(t) + " " + str(reward))
#     x.append(observation[0][0])
#     y.append(observation[0][1])
    

# plt.plot(x, y)
# plt.xlim(-10, 50)
# plt.ylim(-10, 30)
# plt.show()
# print(env.render())
# env.close()

######### IMPLEMENTING REINFORCE ALGORITHM ####################

debug = True

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logistic_regression(s, theta):
    prob_accel = expit(np.dot(s, np.transpose(theta))) #had overflows with my own implementation of sigmoid, found this on the web
    return prob_accel

def draw_action(s, theta):
    prob_accel = logistic_regression(s, theta)
    r = np.random.random()
    if r < prob_accel:
        return 1
    else:
        return 0

# Define the constants of the REINFORCE algortihm

EP_DURATION = 150
ALPHA_INIT = 0.1
NUM_EP = 25
STAND = 0
ACCEL = 1

RENDER = 25

file = open("settingsR.txt", 'w')
settings = ['\nEP_DURATION  = ' + str(EP_DURATION),'\nALPHA_INIT  = ' + str(ALPHA_INIT), '\nNUM_EP  = ' + str(NUM_EP),'\nSTAND  = ' + str(STAND),'\nACCEL  = ' + str(ACCEL),'\nRENDER  = ' + str(RENDER)]
file.close()

######### PLAYING EPISODES FUNCTIONS ########

def play_one_episode(env, theta, max_episode_length=EP_DURATION):

    s_t = env.reset()
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_states.append(s_t)

    for t in range(max_episode_length):

        a_t = draw_action(s_t, theta)
        # a_t = 1
        if(debug):
            print("drawed action " + str(a_t))
        s_t, r_t, done, info = env.step(a_t)

        episode_states.append(s_t)
        episode_actions.append(a_t)
        episode_rewards.append(r_t)

        if done:
            break

    return episode_states, episode_actions, episode_rewards, done


def score_on_multiple_episodes(env, theta, render, num_episodes=NUM_EP, max_episode_length=EP_DURATION):
    
    num_success = 0
    average_return = 0
    num_consecutive_success = [0]

    if(debug):
        print("entering score multiple")

    for episode_index in range(num_episodes):
 
        _, _, episode_rewards, done = play_one_episode(env, theta, max_episode_length)
        success_for_one_try = (np.max(episode_rewards) == 200)
        if(debug):
            print("episode played " + str(episode_index) + ' success = ' + str(success_for_one_try))
        total_rewards = sum(episode_rewards)

        if success_for_one_try:
            num_success += 1
            num_consecutive_success[-1] += 1
        else:
            num_consecutive_success.append(0)

        average_return += (1.0 / num_episodes) * total_rewards

        if(render==True):
            print("Test Episode {0}: Total Reward = {1} - Success = {2}".format(episode_index,total_rewards,success_for_one_try))
            

    if max(num_consecutive_success) >= 10:  
        success = True
    else:
        success = False
        

    return success, num_success, average_return


def compute_policy_gradient(episode_states, episode_actions, episode_rewards, theta):

    H = len(episode_rewards)
    PG = 0

    for t in range(H):

        prob_accel = logistic_regression(episode_states[t], theta)
        a_t = episode_actions[t]
        R_t = sum(episode_rewards[t::])
        if a_t == STAND:
            g_theta_log_pi = - prob_accel * episode_states[t] * R_t
        else:
            prob_stand = (1 - prob_accel)
            g_theta_log_pi = prob_stand * episode_states[t] * R_t

        PG += g_theta_log_pi

    return PG


########### TRAINING FUNCTION ############

def train(env, theta_init, render, max_episode_length = EP_DURATION, alpha_init = ALPHA_INIT):

    theta = theta_init
    episode_index = 0
    average_returns = []
    if(debug):
        print("Render = " +str(render))

    if(render):
        theta_array = []

    success, _, R = score_on_multiple_episodes(env, theta, render)
    if(render): 
        theta_array.append(theta)
        print("new theta stored")

    average_returns.append(R)

    if(debug):
        print("before while")

    # Train until success
    while (not success):

        # Rollout
        episode_states, episode_actions, episode_rewards, done = play_one_episode(env, theta, max_episode_length)
        if(debug):
            print("episode rollout played")
        # Schedule step size
        #alpha = alpha_init
        alpha = alpha_init / (1 + episode_index)

        # Compute gradient
        PG = compute_policy_gradient(episode_states, episode_actions, episode_rewards, theta)
        if(debug):
            print('PG done')
        # Do gradient ascent
        theta += alpha * PG

        # Test new policy
        success, _, R = score_on_multiple_episodes(env, theta, render)

        if(render and ((NUM_EP*episode_index)%RENDER) == 0):  
                theta_array.append(theta)
                print("new theta stored")
        # Monitoring
        average_returns.append(R)

        episode_index += 1

        print("Episode {0}, average return: {1}".format(episode_index, R))

    return theta, episode_index, average_returns, theta_array



############ TRAIN #############

dim = 5 #dimension of observation space

# Init parameters to random
theta_init = np.random.randn(1, dim)

#Train the agent
print('beginning TRAINING')

I_WANT_TO_RENDER = True

theta, i, average_returns, theta_array = train(env, theta_init, I_WANT_TO_RENDER)
print("Solved after {} iterations".format(i))



# Writing the .npy with the values of thetas for the demo if I WANT TO RENDER set to True
if(I_WANT_TO_RENDER):
    np.save("thetas.npy", theta_array)

# Show training curve
plt.plot(range(len(average_returns)),average_returns)
plt.title("Average reward on " + str(NUM_EP) +" episodes")
plt.xlabel("Training Steps")
plt.ylabel("Reward")
plt.savefig("plot_REINFORCE")
plt.show()

env.close()