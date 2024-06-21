import numpy as np
import matplotlib.pyplot as plt

# Define reward distributions
def reward_arm_1():
    return np.random.normal(2, 1)

def reward_arm_2():
    return np.random.choice([5, -6])

def reward_arm_3():
    return np.random.poisson(2)

def reward_arm_4():
    return np.random.exponential(3)

def reward_arm_5():
    arm = np.random.choice([1, 2, 3, 4])
    if arm == 1:
        return reward_arm_1()
    elif arm == 2:
        return reward_arm_2()
    elif arm == 3:
        return reward_arm_3()
    else:
        return reward_arm_4()

# Define the epsilon-greedy agent
class EpsilonGreedyAgent:
    def __init__(self, epsilon, num_arms=5):
        self.epsilon = epsilon
        self.num_arms = num_arms
        self.estimates = np.zeros(num_arms)
        self.action_counts = np.zeros(num_arms)
    
    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.estimates)
    
    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        alpha = 1 / self.action_counts[action]
        self.estimates[action] += alpha * (reward - self.estimates[action])

# Simulation parameters
num_episodes = 1000
episode_length = 100
epsilons = [0.1, 0.01, 0]

# Function to simulate episodes
def simulate(agent):
    rewards_per_episode = np.zeros(num_episodes)
    for episode in range(num_episodes):
        total_reward = 0
        for t in range(episode_length):
            action = agent.select_action()
            if action == 0:
                reward = reward_arm_1()
            elif action == 1:
                reward = reward_arm_2()
            elif action == 2:
                reward = reward_arm_3()
            elif action == 3:
                reward = reward_arm_4()
            elif action == 4:
                reward = reward_arm_5()
            agent.update_estimates(action, reward)
            total_reward += reward
        rewards_per_episode[episode] = total_reward
    return rewards_per_episode

# Run the simulation for each epsilon value
rewards = {}
for epsilon in epsilons:
    agent = EpsilonGreedyAgent(epsilon)
    rewards[epsilon] = simulate(agent)

# Plot the results
plt.figure(figsize=(10, 6))
for epsilon, reward in rewards.items():
    plt.plot(reward, label=f'epsilon = {epsilon}')
plt.xlabel('Episode')
plt.ylabel('Reward at the end of episode')
plt.title('Epsilon-Greedy Performance')
plt.legend()
plt.show()