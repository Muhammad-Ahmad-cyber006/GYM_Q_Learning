import numpy as np
from common import make_env,save_progress,load_model,run_cli

PKL='frozen_lake8x8.pkl'
PNG='frozen_lake8x8.png'

# Reinforcement Learning:
# Machine learning where an agent learns through actions and rewards.

# Environment:
# The world where the agent performs actions and observes results.

# State:
# The current position of the agent.

# Action:
# A move the agent can perform.

# Reward:
# Feedback received after performing an action.

# Q-Learning:
# An algorithm that learns the value of each action in each state.

# Q-Table:
# Stores the estimated value of every state-action pair.
# FrozenLake 8x8 has 64 states and 4 actions.
# Therefore, the Q-Table has 64 rows and 4 columns.

def train(render=False,episodes=17000):
    # Create the 8x8 FrozenLake environment.
    # is_slippery=True means movement may be unpredictable.
    env=make_env('FrozenLake-v1',render,map_name='8x8',is_slippery=True)

    # Initialize Q-Table with zeros.
    # Rows = states, columns = actions.
    q=np.zeros((env.observation_space.n,env.action_space.n))

    # Learning Rate (Alpha):
    # Controls how much new information changes the old Q-value.
    lr=0.9

    # Discount Factor (Gamma):
    # Controls the importance of future rewards.
    # 0 = immediate rewards only, 1 = maximum future importance.
    gamma=0.9

    # Epsilon:
    # Controls exploration. Starts at 1, meaning random actions.
    epsilon=1

    # Epsilon Decay:
    # Gradually decreases exploration during training.
    decay=0.0001

    # Random Number Generator:
    # Generates random numbers for action selection.
    rng=np.random.default_rng()

    # Stores whether the goal was reached in each episode.
    rewards_per_episode=np.zeros(episodes)
    done=0

    try:
        for i in range(episodes):
            # Reset environment. Agent starts at state 0.
            state=env.reset()[0]

            # Terminated = game ended.
            # Truncated = time limit reached.
            terminated=truncated=False
            reward=0

            while not (terminated or truncated):
                # Epsilon-Greedy Strategy:
                # Exploration = choose a random action.
                # Exploitation = choose the action with the highest Q-value.
                if rng.random()<epsilon:
                    action=env.action_space.sample()
                else:
                    action=int(np.argmax(q[state,:]))

                # Perform action and receive the result.
                # new_state = next state, reward = feedback.
                new_state,reward,terminated,truncated,_=env.step(action)

                # Q-Learning Update:
                # Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
                # s = current state, a = selected action.
                # s' = next state, alpha = learning rate.
                # gamma = discount factor.
                # max Q(s',a') = highest estimated future reward.
                q[state,action]+=lr*(reward+gamma*np.max(q[new_state,:])-q[state,action])

                # Move to the next state.
                state=new_state

            # Decrease exploration after each episode.
            epsilon=max(epsilon-decay,0)

            # Reduce learning rate after exploration finishes.
            if epsilon==0:
                lr=0.0001

            # Store 1 if the goal is reached; otherwise store 0.
            if reward==1:
                rewards_per_episode[i]=1

            done=i+1

            # Display progress after every 1000 episodes.
            if i%1000==0:
                recent=rewards_per_episode[max(0,i-99):i+1].sum()
                print(f'Episode: {i}  Epsilon: {epsilon:0.2f}  Goals in last 100: {recent:.0f}')

    finally:
        # Close environment and save training progress.
        save_progress(q,PKL,PNG,rewards_per_episode[:done],how='sum')
        env.close()

# Play Function:
# Loads the trained Q-Table and plays without learning.
def play(episodes=10):
    q=load_model(PKL)
    env=make_env('FrozenLake-v1',True,map_name='8x8',is_slippery=True)

    try:
        wins=0

        for i in range(episodes):
            # Start a new episode.
            state=env.reset()[0]
            terminated=truncated=False
            reward=0

            while not (terminated or truncated):
                # Greedy Policy:
                # Always choose the action with the highest Q-value.
                action=int(np.argmax(q[state,:]))
                state,reward,terminated,truncated,_=env.step(action)

            # Count successful episodes.
            wins+=int(reward==1)
            print(f'Demo episode {i}: {"reached goal" if reward==1 else "failed"}')

        print(f'Goal reached {wins}/{episodes} times.')

    finally:
        env.close()

# Main Function:
# run_cli decides whether to train or play based on command-line arguments.
if __name__=='__main__':
    run_cli(train,play,PKL)