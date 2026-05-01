import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pygame
import matplotlib.pyplot as plt
from torch.distributions import Categorical

#env_render = gym.make('CartPole-v1', render_mode='human')
#env = gym.make('CartPole-v1')
#print(env.reset())

#togliere env tanto non viene utilizzato
def select_action(obs, policy):
    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))

# Utility to compute the discounted total reward.
def compute_returns(rewards, gamma):
    returns = []
    running_return = 0.0

    #Gt = Rt + gamma * G(t+1)
    #Parto calcolando il return dal fondo e running_return é sempre G(t+1)
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)

    returns.reverse()
    return np.asarray(returns)

# Given an environment and a policy, run it up to the maximum number of steps.
def run_episode(env, policy, maxlen=500, device = "cpu"):
    # Collect just about everything.
    observations = []
    actions = []
    log_probs = []
    rewards = []
    
    # Reset the environment and start the episode.
    (obs, info) = env.reset()
    for i in range(maxlen):
        # Get the current observation, run the policy and select an action.
        obs_d = torch.as_tensor(obs, dtype=torch.float32).to(device)
        (action, log_prob) = select_action(obs_d, policy)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        
        # Advance the episode by executing the selected action.
        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return (observations, actions, torch.cat(log_probs), rewards)

class PolicyNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, env.action_space.n)
        
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.softmax(self.fc2(s), dim=-1)
        return s
    
def evaluate_agent(env, policy, M, device):
    total_rewards = 0.0
    episodes_lenghts = 0.0
    policy.eval()
    for _ in range(M):
        (observations, actions, log_probs, rewards) = run_episode(env, policy, device= device)
        total_rewards += sum(rewards)
        episodes_lenghts += len(rewards)
    avg_rewards = total_rewards / M
    avg_lenght = episodes_lenghts / M
    return avg_rewards, avg_lenght
    
def reinforce(policy, env, N, M, env_render=None, gamma=0.99, num_episodes=10, baseline = False, device = "cpu"):
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

    # Track episode rewards in a list.
    running_rewards = [0.0]
    avg_total_rewards = []
    avg_episode_length = []
    
    # The main training loop.
    policy.train()
    for episode in range(num_episodes):
        # Run an episode of the environment, collect everything needed for policy update.
        (observations, actions, log_probs, rewards) = run_episode(env, policy, device =device)
        
        # Compute the discounted reward for every step of the episode. 
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32).to(device)

        if not episode % N:
            avg_rewards, avg_lenght = evaluate_agent(env, policy, M, device)
            avg_total_rewards.append(avg_rewards)
            avg_episode_length.append(avg_lenght)
            print(f'Average total reward: {avg_rewards }')
            print(f'Average episode lenght: {avg_lenght}')
            policy.train()
            
        
        # Keep a running average of total discounted rewards for the whole episode.
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])
        
        # Standardize returns.
        if baseline:
            returns = (returns - returns.mean()) / returns.std()
        
        # Make an optimization step
        opt.zero_grad()
        loss = (-log_probs * returns).mean()
        loss.backward()
        opt.step()
        
        # Render an episode after every 100 policy updates.
        if not episode % 100:
            if env_render:
                policy.eval()
                run_episode(env_render, policy, device=device)
                policy.train()
            print(f'Running reward: {running_rewards[-1]}')
    
    # Return the running rewards.
    policy.eval()
    return running_rewards, avg_total_rewards, avg_episode_length

def main():
    # env_render = gym.make('CartPole-v1', render_mode='human')

    # # Make a policy network and run a few episodes to see how well random initialization works.
    # policy = PolicyNet(env_render)
    # for _ in range(10):
    #     run_episode(env_render, policy)
    
    # # If you don't close the environment, the PyGame window stays visible.
    # env_render.close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seed = 2112
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    env_render = None # gym.make('CartPole-v1', render_mode='human')
    N = 100
    M = 20
    baseline = True

    # Make a policy network.
    policy = PolicyNet(env).to(device)

    # Train the agent.
    running_rewards, avg_rewards, avg_lengths = reinforce(policy, env, N, M, env_render, num_episodes=1000, baseline= baseline, device= device)
    plt.figure()
    plt.plot(running_rewards)
    plt.title('Running Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('running_reward_no_baseline_50.png')

    plt.figure()
    plt.plot(avg_rewards)
    plt.title('Average Total Reward')
    plt.xlabel('Evaluation (every N episodes)')
    plt.ylabel('Avg Reward')
    plt.savefig('avg_reward_no_baseline_50.png')

    plt.figure()
    plt.plot(avg_lengths)
    plt.title('Average Episode Length')
    plt.xlabel('Evaluation (every N episodes)')
    plt.ylabel('Avg Length')
    plt.savefig('avg_length_no_baseline_50.png') 
    # Close up everything
    #env_render.close()
    env.close()

    ren_seed = 1234
    env_render = gym.make('CartPole-v1', render_mode='human')
    env_render.reset(seed=seed)
    avg_reward, avg_length = evaluate_agent(env_render, policy, 10, device)
    env_render.close()
    print(f'Average total reward: {avg_reward }')
    print(f'Average episode lenght: {avg_length}')


if __name__ == "__main__":
    main()