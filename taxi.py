import gymnasium as gym
import numpy as np
from common import make_env,save_progress,load_model,run_cli

PKL='taxi.pkl'
PNG='taxi.png'

# Environment ID:
# Gymnasium versions may support different Taxi environment names.
# Newer versions use Taxi-v4; older versions may use Taxi-v3.
# gym.registry contains the environments supported by the installed version.
ENV_ID='Taxi-v4' if 'Taxi-v4' in gym.registry else 'Taxi-v3'

def train(render=False,episodes=15000):
    # Create Taxi environment.
    env=make_env(ENV_ID,render)

    # Initialize Q-Table with zeros.
    # Taxi has 500 states and 6 actions, so Q-Table = 500 x 6.
    # No discretization is needed because Taxi already provides integer states (0-499).
    q=np.zeros((env.observation_space.n,env.action_space.n))

    # Learning Rate (Alpha): Controls how much new information changes old Q-values.
    lr=0.9

    # Discount Factor (Gamma): Controls the importance of future rewards.
    gamma=0.9

    # Epsilon: Controls exploration. Starts at 1 (100% random actions).
    epsilon=1

    # Epsilon Decay: Gradually reduces exploration.
    # With decay=0.0001, epsilon reaches zero after 10000 episodes.
    decay=0.0001

    rng=np.random.default_rng()

    # Stores total reward received in each episode.
    rewards_per_episode=np.zeros(episodes)
    done=0

    try:
        for i in range(episodes):
            # Reset environment. Taxi gets a passenger and destination.
            state=env.reset()[0]

            # Terminated = successful passenger drop-off.
            # Truncated = episode reaches the time limit.
            terminated=truncated=False
            rewards=0

            while not (terminated or truncated):
                # Epsilon-Greedy Strategy:
                # Exploration = random action.
                # Exploitation = action with the highest Q-value.
                if rng.random()<epsilon:
                    action=env.action_space.sample()
                else:
                    action=int(np.argmax(q[state,:]))

                # Perform action and receive the result.
                new_state,reward,terminated,truncated,_=env.step(action)

                # Add reward to the episode's total.
                rewards+=reward

                # Q-Learning / Bellman Update:
                # Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
                # Same formula as FrozenLake and MountainCar.
                # Only the state representation is different.
                q[state,action]+=lr*(reward+gamma*np.max(q[new_state,:])-q[state,action])

                # Move to the next state.
                state=new_state

            # Reduce exploration after every episode.
            epsilon=max(epsilon-decay,0)

            # Reduce learning rate after exploration finishes.
            if epsilon==0:
                lr=0.0001

            # Store total episode reward.
            rewards_per_episode[i]=rewards
            done=i+1

            # Display average reward every 1000 episodes.
            if i%1000==0:
                mean=rewards_per_episode[max(0,i-99):i+1].mean()
                print(f'Episode: {i}  Epsilon: {epsilon:0.2f}  Mean(last100): {mean:0.1f}')

    finally:
        # Save training progress and close environment.
        try:
            save_progress(q,PKL,PNG,rewards_per_episode[:done],how='sum')
        finally:
            env.close()

def play(episodes=10):
    # Load the trained Q-Table.
    q=load_model(PKL)

    # Create Taxi environment with rendering enabled.
    env=make_env(ENV_ID,render=True)

    try:
        for i in range(episodes):
            # Reset environment.
            state=env.reset()[0]
            terminated=truncated=False
            rewards=0

            while not (terminated or truncated):
                # Greedy Policy: Always choose the action with the highest Q-value.
                action=int(np.argmax(q[state,:]))

                # Perform action and receive the result.
                state,reward,terminated,truncated,_=env.step(action)
                rewards+=reward

            print(f'Demo episode {i}: reward {rewards}')

    finally:
        # Close environment.
        env.close()

if __name__=='__main__':
    run_cli(train,play,PKL)