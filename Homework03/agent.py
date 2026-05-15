import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, frame_stack_num=4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=frame_stack_num, out_channels=32, kernel_size=8, stride=4)),
            nn.ReLU(),  
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(in_features= 64 * 7 * 7, out_features=512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        #immagine normalizzata tra 0 e 1
        return self.critic(self.network(x / 255.0))
    
    def get_value_and_action(self, x, action=None):
        hidden = self.network(x / 255.0) #immagine normalizzata
        dist = Categorical(logits= self.actor(hidden))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden)