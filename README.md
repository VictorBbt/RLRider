**RIDER LEARNING:** 

Reinforcement learning algorithms (DQN, Reinforce) applied to Rider game

In this project, we have created a Unity game with the same mechanics at the Ketchapp’s game Rider. We then wrapped up the Unity environment in a gym environment to implement our own RL algorithms, mainly Deep Q-Learning (using PyTorch) and REINFORCE algorithm (gradient descent) to compare performances.

The main challenges were the binding of the Unity Environment to a gym environmnent, how to solve the computational approximations in the calculus of the gradient descent, and how to fine-tune the DQN.

In a future work, we first hope to compare our algorithms with those of Stable Baselines 3. The possible prolongations of this work include:

- Add a reward for style, i.e. give a reward for flips
- Implement a network with memory by adding a recurrent layer (R2D2 algorithm)


How to install:

- **Windows (10 or higher)**: download the file ‘*WindowsBuild*’ on your computer. In this file, click on the ‘*.exe*’ file to open it and click ‘continue’ if a warning pops up. You can then play the game with left-clicking on your mouse. Then, in the same directory as ‘*WindowsBuild*’, download the files ‘*Reinforce.ipynb*’ and ‘*DQN.ipynb*’ and follow the instructions to run the code.

- **Mac (10.5 or higher)**: download the file ‘MacBuild.zip’ on your computer. Open it to uncompress it.? Once you have a file ‘*MacBuild*’, open it and right click on the ‘*.app*’ file and click Open. You can then play the game with left-clicking on your mouse. Then, in the same directory as ‘*WindowsBuild*’, download the files ‘*Reinforce.ipynb*’ and ‘*DQN.ipynb*’ and follow the instructions to run the code.

- **Linux:** download the file ‘LinuxBuild’ on your computer. Then, go to this file in your terminal (‘*> cd YOURPATH/LinuxBuild*’). Run the following instruction ‘*> chmod .x FILE.x86\_64*’, and then ‘*> ./FILE.x86\_64*’. You can then play the game with left-clicking on your mouse. Then, in the same directory as ‘*WindowsBuild*’, download the files ‘*Reinforce.ipynb*’ and ‘*DQN.ipynb*’ and follow the instructions to run the code.

Now you should be good tu run the project by following the instructions of the ‘*.ipynb*’ files. Enjoy!

Credits:

- Clément GARANCINI: githublink
- Marc-Antoine Oudotte: githublink

