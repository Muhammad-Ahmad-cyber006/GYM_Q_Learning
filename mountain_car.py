import numpy as np
from common import make_env,save_progress,load_model,run_cli

PKL='mountain_car.pkl'
PNG='mountain_car.png'

# DIVISIONS: Number of cut points used to divide continuous values into buckets.
# MountainCar has two continuous values: position and velocity.
# We divide both into 20 cut points to use them in a Q-Table.
DIVISIONS=20

def make_bins(env):
    # Discretization: Converting continuous values into discrete buckets.
    lo,hi=env.observation_space.low,env.observation_space.high

    # np.linspace(): Generates equally spaced values between minimum and maximum.
    pos_space=np.linspace(lo[0],hi[0],DIVISIONS)
    vel_space=np.linspace(lo[1],hi[1],DIVISIONS)
    return pos_space,vel_space

def train(render=False,episodes=15000):
    env=make_env('MountainCar-v0',render)
    # Create buckets for position and velocity.
    pos_space,vel_space=make_bins(env)

    # np.digitize(): Converts continuous values into bucket numbers.
    # Q-Table = Position buckets x Velocity buckets x Actions.
    # 21 x 21 x 3 = 1323 Q-values.
    shape=(len(pos_space)+1,len(vel_space)+1,env.action_space.n)
    q=np.zeros(shape)

    # Learning Rate (Alpha): Controls how much new information changes old Q-values.
    lr=0.9
    # Discount Factor (Gamma): Controls the importance of future rewards.
    gamma=0.9
    # Epsilon: Controls random exploration. Initially 1 means 100% random actions.
    epsilon=1
    # Epsilon Decay: Reduces random exploration after every episode.
    # Epsilon reaches zero halfway through training.
    decay=2/episodes
    rng=np.random.default_rng()
    rewards_per_episode=np.zeros(episodes)
    done=0

    try:
        for i in range(episodes):
            # Reset environment and get initial position and velocity.
            state=env.reset()[0]

            # Convert continuous position and velocity into bucket numbers.
            p=int(np.digitize(state[0],pos_space))
            v=int(np.digitize(state[1],vel_space))
            terminated=False
            rewards=0

            # Normally MountainCar has a 200-step limit.
            # We ignore truncation and allow up to 1000 steps.
            while not terminated and rewards>-1000:
                # Epsilon-Greedy: Explore randomly or exploit the best-known action.
                if rng.random()<epsilon:
                    action=env.action_space.sample()
                else:
                    action=int(np.argmax(q[p,v,:]))

                # Perform action and get the new state and reward.
                new_state,reward,terminated,_,_=env.step(action)

                # Convert new position and velocity into buckets.
                new_p=int(np.digitize(new_state[0],pos_space))
                new_v=int(np.digitize(new_state[1],vel_space))

                # Q-Learning Update: Same formula as FrozenLake.
                # Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
                # Here we use both position and velocity to access the Q-Table.
                q[p,v,action]+=lr*(reward+gamma*np.max(q[new_p,new_v,:])-q[p,v,action])

                # Move to the new position and velocity buckets.
                p,v=new_p,new_v
                rewards+=reward

            # Reduce epsilon to decrease exploration and increase exploitation.
            epsilon=max(epsilon-decay,0)
            rewards_per_episode[i]=rewards
            done=i+1

            # Display average reward after every 100 episodes.
            if i%100==0:
                mean=rewards_per_episode[max(0,i-99):i+1].mean()
                print(f'Episode: {i}  Epsilon: {epsilon:0.2f}  Mean(last100): {mean:0.1f}')
    finally:
        # Close environment and save the trained Q-Table.
        env.close()
        save_progress(q,PKL,PNG,rewards_per_episode[:done])

def play(episodes=10):
    # Load the trained Q-Table.
    q=load_model(PKL)
    env=make_env('MountainCar-v0',render=True)

    # Use the same discretization cut points as training.
    pos_space,vel_space=make_bins(env)
    try:
        for i in range(episodes):
            state=env.reset()[0]

            # Convert initial position and velocity into buckets.
            p=int(np.digitize(state[0],pos_space))
            v=int(np.digitize(state[1],vel_space))
            terminated=False
            rewards=0

            while not terminated and rewards>-1000:
                # Greedy Policy: Always choose the action with the highest Q-value.
                action=int(np.argmax(q[p,v,:]))

                # Perform action and get the new state.
                state,reward,terminated,_,_=env.step(action)

                # Convert new position and velocity into buckets.
                p=int(np.digitize(state[0],pos_space))
                v=int(np.digitize(state[1],vel_space))
                rewards+=reward

            print(f'Demo episode {i}: reward {rewards:.0f}  ({"reached flag" if terminated else "gave up"})')
    finally:
        env.close()

if __name__=='__main__':
    run_cli(train,play,PKL)