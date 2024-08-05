# Creating a Custom Environment for a hospital set up using RL
1. Define Actions and Rewards
Actions: The agents can perform two actions:

1.1. Move to a different room escaping obstacles in the way.
    1. Up
    2. Down
    3. Left
    4. Right

Rewards: The agent receives a reward of +1 for reaching the destination and -1 for hitting an obstacle.

2. Define States
The states are the rooms in the hospital. The agent can be in any of the rooms at any given time.

3. Define the Environment
The environment is the hospital set up with rooms and obstacles.

4. Define the Agent
The agent is the person who is trying to reach the destination.

5. Define the Policy
The policy is the strategy that the agent uses to reach the destination.

6. Define the Q-Table
The Q-Table is a table that stores the Q-values for each state-action pair.

7. Define the Q-Learning Algorithm
The Q-Learning algorithm is used to update the Q-values in the Q-Table based on the rewards received by the agent.

8. Train the Agent
The agent is trained by running the Q-Learning algorithm for a number of episodes.

9. Test the Agent
The agent is tested by running the Q-Learning algorithm for a number of episodes and observing the rewards received by the agent.

10. Evaluate the Agent
The agent is evaluated by comparing the rewards received by the agent during training and testing.

11. Improve the Agent
The agent can be improved by changing the policy, the Q-Table, or the Q-Learning algorithm.


# Creating a Custom Environment for a hospital set up using RL
## Define Actions and Rewards
Actions: The agents can perform two actions:
- Move to a different room escaping obstacles in the way.
    - Up
    - Down
    - Left
    - Right

Rewards: The agent receives a reward of +1 for reaching the destination and -1 for hitting an obstacle.

## Define States
The states are the rooms in the hospital. The agent can be in any of the rooms at any given time.

## Define the Environment
The environment is the hospital set up with rooms and obstacles.

## Define the Agent
The agent is the person who is trying to reach the destination.

# Setup Instructions
    Clone the repository: ```git clone <repository_link>```
    Install dependencies: ```pip install -r requirements.txt```
    Run the training script: ```python train.py```
    Test the trained model: ```python play.py```


