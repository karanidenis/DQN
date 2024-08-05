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
1. Clone the repository: `git clone https://github.com/karanidenis/DQN`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training script: `python train.py`
4. Test the trained model: `python play.py`


