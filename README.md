# RIDER LEARNING

**Reinforcement learning algorithms (DQN, Reinforce) applied to Rider game**

In this project, we have created a Unity game with the same mechanics at the Ketchapp’s game Rider. We then wrapped up the Unity environment in a gym environment to implement our own RL algorithms, mainly Deep Q-Learning (using PyTorch) and REINFORCE algorithm (gradient descent) to compare performances.

The main challenges were the binding of the Unity Environment to a gym environmnent, how to solve the computational approximations in the calculus of the gradient descent, and how to fine-tune the DQN.

In a future work, we first hope to compare our algorithms with those of Stable Baselines 3. The possible prolongations of this work include:

- Add a reward for style, i.e. give a reward for flips
- Implement a network with memory by adding a recurrent layer (R2D2 algorithm)


## How to install:

The build files are available for download in the following drive:
https://drive.google.com/drive/folders/1dhv1YaP4PuzNex9NTjvizuL6usFsuz0w?usp=sharing

- **Windows (10 or higher)**: download the folder ‘*Windows*’ on your computer. In this folder you will find three folder, one for each level of the game. In each of these folders, you will find several files including an executable. Click on the ‘*.exe*’ file to open it and click ‘continue’ if a warning pops up. You can then play the game with left-clicking on your mouse. Then, in the same folder as this executable, download the files ‘*Reinforce.py*’ and ‘*DQN.py*’ and follow the instructions to run the code.

- **Mac (10.5 or higher)**: download the folder ‘*Mac*’ on your computer. In this folder you will find three compressed file, one for each level of the game. You can decompress them. In each of these folders, you will find several files including an executable. Click on the ‘*.app*’ file to open it and click ‘continue’ if a warning pops up. You can then play the game with left-clicking on your mouse. Then, in the same folder as this executable, download the files ‘*Reinforce.py*’ and ‘*DQN.py*’ and follow the instructions to run the code.

- **Linux:** download the folder ‘*Linux*’ on your computer. In this folder you will find three folder, one for each level of the game. In each of these folders, you will find several files including an executable. Click on the ‘*.x86_64*’ file to open it and click ‘continue’ if a warning pops up. You can then play the game with left-clicking on your mouse. Then, in the same folder as this executable, download the files ‘*Reinforce.py*’ and ‘*DQN.py*’ and follow the instructions to run the code.

Now you should be good tu run the project by following the instructions of the ‘*.py*’ files. Enjoy!

## To go further

Were you to delve further into Deep Reinforcement Learning Algorithms, you cand find notebooks with implementations of **value-based algorithms (DQN, DDQN, DQN PER,...)** [here](https://github.com/VictorBbt/DQN-Pytorch/tree/main), and another with **policy-gradient algorithms (REINFORCE with or without baseline, continuous or discrete, actor critic)** [here](https://github.com/VictorBbt/Policy-Gradient-Methods/tree/main).

Credits:

Clément Garancini
Victor Barberteguy
Marc-Antoine Oudotte
