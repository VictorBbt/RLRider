###### IMPLEMENTS THE DQN #########

# Libraries
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import copy
from tqdm.notebook import tqdm
import random

# WRAP THE ENVIRONMENT

# Input the level

LEVEL = 3
uEnv = UnityEnvironment("/Users/victorbarberteguy/Desktop/3A/INF581/RL/Rider/Build_Level"+str(LEVEL)+".app", worker_id=0, seed=1, no_graphics=False)
env = UnityToGymWrapper(uEnv, uint8_visual = False, flatten_branched= True,  allow_multiple_obs=True)

debug = True
# Constants

EPISODES = 200
LR = 0.001
MEM_SIZE = 10000
BATCH_SIZE = 72
GAMMA = 0.97
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
sync_freq = 5

def dynamicLearningRate(LR, decay = 0.95, LRmin = 0.0001):
    '''Modifies the learning rate with a decay in order to explore increasingly less as the agent gets better'''
    LR = LR*decay
    LR = np.clip(LR, LRmin, 1)

    return LR

file = open("settingsDQN.txt", 'w')
settings = ['\nEPISODES  = ' + str(EPISODES),'\nLR  = ' + str(LR), '\nMEM_SIZE  = ' + str(MEM_SIZE),'\nBATCH_SIZE  = ' + str(BATCH_SIZE),'\nGAMMA  = ' + str(GAMMA),'\nEXPLORATION_MAX  = ' + str(EXPLORATION_MAX),'\nEXPLORATION_DECAY  = ' + str(EXPLORATION_DECAY),'\nEXPLORATION_MIN  = ' + str(EXPLORATION_MIN),'\nsync_freq = ' + str(sync_freq) ]
file.writelines(settings)
file.close()

# Define the network

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = 5
        self.action_space = 2

        self.fc1 = nn.Linear(self.input_shape, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss = nn.MSELoss()
        #self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Replay Buffer

class ReplayBuffer:
    def __init__(self):
        self.memory = deque(maxlen=MEM_SIZE)
    
    def add(self, experience):
        self.memory.append(experience)
    
    def sample(self):
        minibatch = random.sample(self.memory, BATCH_SIZE)

        state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch])
        action_batch = torch.tensor([a for (s1,a,r,s2,d) in minibatch])
        reward_batch = torch.tensor([r for (s1,a,r,s2,d) in minibatch])
        state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch])
        done_batch = torch.tensor([d for (s1,a,r,s2,d) in minibatch])

        return (state1_batch, action_batch, reward_batch, state2_batch, done_batch)
    
# Implement the DQN

class DQN:
    def __init__(self):
        self.replay = ReplayBuffer() 
        self.exploration_rate = EXPLORATION_MAX
        self.network = Network()
        self.network2 = copy.deepcopy(self.network) #A
        self.network2.load_state_dict(self.network.state_dict())


    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            return env.action_space.sample()

        # Convert observation to PyTorch Tensor
        state = torch.tensor(observation).float().detach()
        #state = state.to(DEVICE)
        state = state.unsqueeze(0)
            
        ### BEGIN SOLUTION ###

        # Get Q(s,.)
        q_values = self.network(state)

        # Choose the action to play
        action = torch.argmax(q_values).item()

        return action

    def learn(self):
        if len(self.replay.memory)< BATCH_SIZE:
            return

        # Sample minibatch s1, a1, r1, s1', done_1, ... , sn, an, rn, sn', done_n
        state1_batch, action_batch, reward_batch, state2_batch, done_batch = self.replay.sample()

        # Compute Q values
        q_values = self.network(state1_batch).squeeze()

        with torch.no_grad():
            # Compute next Q values
            next_q_values = self.network2(state2_batch).squeeze()

        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        predicted_value_of_now = q_values[batch_indices, action_batch]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]

        # Compute the q_target
        q_target = reward_batch + GAMMA * predicted_value_of_future * (1-(done_batch).long())

        # Compute the loss (c.f. self.network.loss())
        loss = self.network.loss(q_target, predicted_value_of_now)

        ### END SOLUTION ###

        # Complute 𝛁Q
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


    def returning_epsilon(self):
        return self.exploration_rate
    

# Training

agent = DQN()

best_reward = 0
average_reward = 0
episode_number = []
average_reward_number = []
success = False
#number of successes on 10 consecutive tries
n_success_10_try = 0

#store the number of successes on 10 tries to plot it
average_success_10_try = []

j=0
dim = 5
if(debug):
    print("BEGINNING TRAINING")
for i in tqdm(range(1, EPISODES+1)):
    if(debug):
            print("episode " + str(i) + " playing...")
    
    state = env.reset()
    state = np.reshape(state, [1, dim])
    score = 0

    while True:
        j+=1

        action = agent.choose_action(state)

        # if(debug):
        #     print("drawed action " + str(action))

        state_, reward, done, info = env.step(action)
        state_ = np.reshape(state_, [1, dim])
        state = torch.tensor(state).float()
        state_ = torch.tensor(state_).float()

        exp = (state, action, reward, state_, done)
        agent.replay.add(exp)
        agent.learn()


        state = state_
        score += reward

        if(reward == 200) :
            success = True
            
            

        if j % sync_freq == 0:
            agent.network2.load_state_dict(agent.network.state_dict())

        if done:
            if score > best_reward:
                best_reward = score

            average_reward += score 
            if(success): #goal reached
                LR = dynamicLearningRate(LR)
                success = False
                n_success_10_try += 1


            if i%10==0:
                print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {} Success = {}/10".format(i, average_reward/i, best_reward, score, agent.returning_epsilon(), n_success_10_try))
                #test_model(agent,10, observation_space)
                print("Learning rate: " + str(LR))
                average_success_10_try.append(n_success_10_try)
                n_success_10_try = 0

            break
            
        
        episode_number.append(i)
        average_reward_number.append(average_reward/i)

plt.subplot(1,2,1)
plt.plot(episode_number, average_reward_number)
plt.title("Plot of the reward")
plt.xlabel('Episodes ')
plt.ylabel('Reward ')

plt.subplot(1, 2, 2) # index 2
x2 = [10*(i+1) for i in range(int(EPISODES/10))]
plt.scatter(x2, average_success_10_try)
plt.plot(x2, [7 for i in range(len(x2))], color = 'r')
plt.ylim(-1,11)
plt.title("Successes each 10 tries")
plt.xlabel('Episodes')
plt.ylabel('N success every 10 tries')

plt.savefig("ResultsDQN")
env.close()
plt.show()