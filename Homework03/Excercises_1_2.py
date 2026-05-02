import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.distributions import Categorical

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

class StateValueNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = self.fc2(s)
        return s

def train_policy(log_probs, returns, opt):
    opt.zero_grad()
    loss = (-log_probs * returns).mean()
    loss.backward()
    opt.step()

def train_value(state_values, returns, opt):
    opt.zero_grad()
    loss = F.mse_loss(state_values, returns)
    loss.backward()
    opt.step()

def evaluate_agent(env, policy, M, device):
    total_rewards = 0.0
    episodes_lenghts = 0.0
    policy.eval()
    with torch.no_grad():
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

        if not episode % N:
            avg_rewards, avg_lenght = evaluate_agent(env, policy, M, device)
            avg_total_rewards.append(avg_rewards)
            avg_episode_length.append(avg_lenght)
            print(f'Average total reward: {avg_rewards }')
            print(f'Average episode lenght: {avg_lenght}')
            policy.train()
            
    
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

def baseline_reinforce(policy, state_val_net, env, N, M, env_render=None, gamma=0.99, num_episodes=10, baseline = False, device = "cpu"):
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    state_opt = torch.optim.Adam(state_val_net.parameters(), lr=1e-2)

    avg_total_rewards = []
    avg_episode_length = []

    policy.train()
    state_val_net.train()

    for episode in range(num_episodes):
        (observations, actions, log_probs, rewards) = run_episode(env, policy, device =device)

        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32).to(device)

        states = torch.tensor(observations, dtype=torch.float32).to(device)

        states_values = state_val_net(states).squeeze(-1)

        train_value(states_values, returns, state_opt)

        advantages = returns - states_values.detach()

        train_policy(log_probs, advantages, opt)

        if not episode % N:
            avg_rewards, avg_lenght = evaluate_agent(env, policy, M, device)
            avg_total_rewards.append(avg_rewards)
            avg_episode_length.append(avg_lenght)
            print(f'Average total reward: {avg_rewards }')
            print(f'Average episode lenght: {avg_lenght}')
            policy.train()

        if not episode % 100:
            if env_render:
                policy.eval()
                run_episode(env_render, policy, device=device)
                policy.train()
        
    policy.eval()
    return avg_total_rewards, avg_episode_length

def save_plot(data, title, xlabel, ylabel, filename, folder="results"):
    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, filename)

    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filepath)
    plt.close()

def setup(seed=2112, env_name='CartPole-v1', render=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    env.reset(seed=seed)

    env_render = None
    if render:
        env_render = gym.make(env_name, render_mode='human')
        env_render.reset(seed=seed)

    return device, env, env_render

def final_evaluation(env_name, policy, device, episodes=10, seed=1234):
    env_render = gym.make(env_name, render_mode='human')
    env_render.reset(seed=seed)

    avg_reward, avg_length = evaluate_agent(env_render, policy, episodes, device)

    env_render.close()

    print(f'Final Average reward: {avg_reward}')
    print(f'Final Average length: {avg_length}')

    return avg_reward, avg_length

def run_experiment(trainer_fn, env_name='CartPole-v1', num_episodes=1000, N=100, M=20, seed=1221, render=False, prefix="", use_baseline=False, std=False, folder="results"):
    # Setup
    device, env, env_render = setup(seed, env_name, render)

    # Policy
    policy = PolicyNet(env).to(device)

    # Value network (solo se serve)
    state_val_net = None
    if use_baseline:
        state_val_net = StateValueNet(env).to(device)

    # Training
    if use_baseline:
        avg_rewards, avg_lengths = trainer_fn(policy, state_val_net, env, N, M, env_render=env_render, num_episodes=num_episodes, device=device)
        running_rewards = None
    else:
        running_rewards, avg_rewards, avg_lengths = trainer_fn(policy, env, N, M, env_render=env_render, num_episodes=num_episodes, baseline= std, device=device)

    # Plot
    if running_rewards is not None:
        save_plot(running_rewards, 'Running Reward', 'Episode', 'Reward', f'{prefix}running_reward.png', folder=folder)
        save_plot(avg_rewards, 'Average Reward', 'Eval step', 'Reward', f'{prefix}avg_reward.png', folder=folder)
        save_plot(avg_lengths, 'Average Length', 'Eval step', 'Length', f'{prefix}avg_length.png', folder=folder)
    else:
        save_plot(avg_rewards, 'Average Reward', 'Eval step', 'Reward', f'{prefix}avg_reward.png', folder=folder)
        save_plot(avg_lengths, 'Average Length', 'Eval step', 'Length', f'{prefix}avg_length.png', folder=folder)

    # Final evaluation
    final_evaluation(env_name, policy, device)

    env.close()

    return policy

def main():
    std = False
    use_baseline = True

    if std:
        prefix = "std_"
    else:
        prefix = "no_std_"
    
    if use_baseline:
        prefix = "baseline_"
        trainer_fn= baseline_reinforce
    else:
        trainer_fn = reinforce

    run_experiment(trainer_fn, prefix = prefix, use_baseline= use_baseline, std=std)

if __name__ == "__main__":
    main()