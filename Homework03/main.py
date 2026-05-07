import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import os
import argparse
import random
import time
from torch.distributions import Categorical
from dataclasses import dataclass

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CarRacing-v3",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track",  action=argparse.BooleanOptionalAction, default=False,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", action=argparse.BooleanOptionalAction, default=False,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", action=argparse.BooleanOptionalAction, default=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", action=argparse.BooleanOptionalAction,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

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
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden)


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, continuous=False, render_mode ="rgb_array")

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 3)

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk

def compute_advantages(agent, buffer, next_done, next_obs, args):
    with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae: #calculate GAE generalized advantage estimation
                advantages = torch.zeros_like(buffer.rewards)
                last_gae = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1: #ultimo step se l'episodio non é terminato stimo il return con il critic altrimenti é 0
                        next_non_terminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        next_non_terminal = 1.0 - buffer.dones[t + 1]
                        next_return = buffer.state_values[t + 1]
                    delta = buffer.rewards[t] + args.gamma * next_return * next_non_terminal - buffer.state_values[t]
                    last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
                    advantages[t] = last_gae
                returns = advantages + buffer.state_values
            else: #one step advantage
                returns = torch.zeros_like(buffer.rewards)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1: #ultimo step se l'episodio non é terminato stimo il return con il critic altrimenti é 0
                        next_non_terminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        next_non_terminal = 1.0 - buffer.dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = buffer.rewards[t] + args.gamma * next_non_terminal * next_return
                advantages = returns - buffer.state_values
    return advantages, returns

def ppo():
    print("hello")

def test():
    envs = gym.vector.SyncVectorEnv(
        [make_env("CarRacing-v3", 1234 + i, i, True, "test") for i in range(2)]
    )
    obs = envs.reset()
    for i in range(100):
        print(i)
        action = envs.action_space.sample()
        obs, reward, term, trunc, info = envs.step(action)
        print(type(obs))
        if "episode" in info:
            done_envs = info["_episode"]
            returns = info["episode"]["r"]

            for i, done in enumerate(done_envs):
                if done:
                    print(f"env {i} episodic return {returns[i]}")
    
    envs.close()

def collect_rollout(envs, agent, buffer, next_obs, next_done, global_step, args, device,):
    for step in range(args.num_steps):
        global_step += args.num_envs

        buffer.obs[step] = next_obs
        buffer.dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_value_and_action(next_obs)
            buffer.values[step] = value.flatten()#value prima di flatten é [8,1] tolgo dimensione inutile

        buffer.actions[step] = action
        buffer.logprobs[step] = logprob

        next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())

        buffer.rewards[step] = torch.tensor(reward).to(device).view(-1) #ugale a value sopra

        done = term | trunc#or logico serve per sapere se l'env ha finito
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.Tensor(done).to(device)

        # logging episodi (lo lasciamo qui per ora)
        if "episode" in info:
            done_envs = info["_episode"]
            returns = info["episode"]["r"]
            lengths = info["episode"]["l"]

            for i, done in enumerate(done_envs):
                if done:
                    print(f"global_step={global_step}, episodic_return={returns[i]}")
                    print(f"global_step={global_step}, episodic_length={lengths[i]}")
                    if args.track:
                        wandb.log(
                            {
                                "episodic_return": returns[i],
                                "episodic_length": lengths[i],
                            },
                            step=global_step,
                        )

    return next_obs, next_done, global_step

def update_agent(agent, optimizer, batch, args,):
    b_inds = np.arange(args.batch_size)

    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)

        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_value_and_action(
                batch.obs[mb_inds],
                batch.actions.long()[mb_inds], #long perché Categorical voule indici int64
            )

            logratio = newlogprob - batch.logprobs[mb_inds]
            ratio = logratio.exp()

            mb_advantages = batch.advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)#per evitare di dividere per zero

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - args.clip_coef, 1 + args.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - batch.returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
    
    #ritorna loss dell'ultima epoca di aggiornamento e dell'ultimo minibatch
    return {"policy_loss": pg_loss.item(), "value_loss": v_loss.item(), "entropy_loss": entropy_loss.item(),}

@dataclass
class PPOBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor

@dataclass
class RolloutBuffer:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor

    # flatten the batch dim del batch = num_steps * num envs
    def to_batch(self, advantages: torch.Tensor, returns: torch.Tensor,) -> PPOBatch:
        return PPOBatch(
            obs=self.obs.reshape((-1,) + self.obs.shape[2:]),
            actions=self.actions.reshape((-1,) + self.actions.shape[2:]),
            logprobs=self.logprobs.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
        )




def main():
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, args.exp_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    buffer = RolloutBuffer(
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device), #torch.Size([128, 8, 3, 96, 96])
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device), #torch.Size([128, 8])
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device),  #torch.Size([128, 8])
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device),  #torch.Size([128, 8])
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device), #torch.Size([128, 8])
        values = torch.zeros((args.num_steps, args.num_envs)).to(device), #torch.Size([128, 8])
    )
    
  
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device) #torch.Size([8, 3, 96, 96])
    next_done = torch.zeros(args.num_envs).to(device) #torch.Size([8])
    num_updates = args.total_timesteps // args.batch_size #9765

    
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        next_obs, next_done, global_step = collect_rollout(
            envs= envs,
            agent= agent,
            buffer= buffer,
            next_obs= next_obs,
            next_done= next_done,
            global_step= global_step,
            args= args,
            device= device,
        )

        advantages, returns = compute_advantages(agent= agent, buffer= buffer, next_done= next_done, next_obs= next_obs, args= args,)   

        batch = buffer.to_batch(advantages= advantages, returns= returns)
      
        # Optimizing the policy and value network
        metrics= update_agent(agent= agent, optimizer= optimizer, batch= batch, args= args,)

        sps = int(global_step / (time.time() - start_time))

        if args.track:
            wandb.log(
                {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                    "SPS": sps,
                },
                step=global_step,
            )


if __name__ == "__main__":
    main()
