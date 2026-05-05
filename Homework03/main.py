import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from torch.distributions import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, frame_stack_num=3):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=frame_stack_num, out_channels=6, kernel_size=7, stride=3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            layer_init(nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4, stride=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            layer_init(nn.Linear(in_features= 12 * 6 * 6, out_features=216)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(216, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(216, 1), std=1)

    def get_value(self, x):
        #immagine normalizzata tra 0 e 1
        return self.critic(self.network(x / 255.0))
    
    def get_value_and_action(self, x, action=None):
        hidden = self.network(x / 255.0) #immagine normalizzata
        dist = Categorical(self.actor(hidden))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy, self.critic(hidden)


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, continuous=False)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 3)

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk



def main():
    print("Hello from homework03!")


if __name__ == "__main__":
    main()
