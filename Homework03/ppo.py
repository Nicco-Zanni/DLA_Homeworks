import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import wandb
import os
import argparse
import random
from dataclasses import dataclass
from agent import Agent
from car_race import make_vec_envs

def parse_args():
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CarRacing-v3",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
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
    parser.add_argument("--save-frequency", type=int, default=50,
        help="save checkpoint every N updates",)
    parser.add_argument("--resume", type=str, default=None,
        help="path to checkpoint")
    parser.add_argument("--capture-video", action=argparse.BooleanOptionalAction, default=False,
        help="weather to capture videos of the agent performances")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1024,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", action=argparse.BooleanOptionalAction, default=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.015,
        help="the target KL divergence threshold")
    parser.add_argument("--render-mode", type=str, default=None, choices=["human", "rgb_array", None],
        help="gymnasium render mode")
    parser.add_argument("--reward-clip", type=float, default=None,
        help="upper bound for reward clipping; if None no clipping is applied")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    if args.render_mode is None:
        args.capture_video = False
  
    return args

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
                        next_return = buffer.values[t + 1]
                    delta = buffer.rewards[t] + args.gamma * next_return * next_non_terminal - buffer.values[t]
                    last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
                    advantages[t] = last_gae
                returns = advantages + buffer.values
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
                advantages = returns - buffer.values
    return advantages, returns

def collect_rollout(envs, agent, buffer, next_obs, next_done, global_step, args, device,):
    episodic_returns = []
    episodic_lengths = []
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

        if "episode" in info:
            done_envs = info["_episode"]
            returns = info["episode"]["r"]
            lengths = info["episode"]["l"]

            for i, done in enumerate(done_envs):
                if done:
                    episodic_returns.append(returns[i])
                    episodic_lengths.append(lengths[i])
    
    avg_ep_return= np.mean(episodic_returns)
    avg_ep_length= np.mean(episodic_lengths)
    print(f"global_step={global_step}, avg_episodic_return={avg_ep_return}")
    print(f"global_step={global_step}, avg_episodic_length={avg_ep_length}")

    return next_obs, next_done, global_step, avg_ep_return, avg_ep_length

def update_agent(agent, optimizer, batch, args,):
    b_inds = np.arange(args.batch_size)
    pg_losses = []
    v_losses = []
    entropy_losses = []
    total_losses = []

    approx_kls = []
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        epoch_kls = []
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_value_and_action(
                batch.obs[mb_inds],
                batch.actions.long()[mb_inds], #long perché Categorical voule indici int64
            )

            logratio = newlogprob - batch.logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    epoch_kls.append(approx_kl.item())

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

            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            entropy_losses.append(entropy_loss.item())
            total_losses.append(loss.item())

            approx_kls.append(approx_kl.item())
        
        mean_kl = np.mean(epoch_kls)
        if args.target_kl is not None:
                if mean_kl > args.target_kl:
                    break
    
    y_pred, y_true = batch.values.cpu().numpy(), batch.returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    
    return {"policy_loss": np.mean(pg_losses), "value_loss": np.mean(v_losses), "entropy_loss": np.mean(entropy_losses),
            "total_loss": np.mean(total_losses), "approx_kl": np.mean(approx_kls),"clipfrac": np.mean(clipfracs),
            "explained_var": explained_var,}

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
            values=self.values.reshape(-1)
        )

def setup(args, run_name):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_vec_envs(args.num_envs,args.gym_id, args.seed, args.render_mode, args.capture_video, args.reward_clip, run_name)
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

    return envs, agent, optimizer, buffer, device

def save_checkpoint(agent, optimizer, global_step, update, args, path,):
    checkpoint = {
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "global_step": global_step,
        "update": update,

        "args": vars(args),
    }

    torch.save(checkpoint, path)

def load_checkpoint(path, agent, optimizer, device):
    checkpoint = torch.load(path, map_location=device)

    agent.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    global_step = checkpoint["global_step"]
    update = checkpoint["update"]

    print(f"Loaded checkpoint from {path}")

    return global_step, update

def main():
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}"
    checkpoint_dir = f"checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.track:

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    envs, agent, optimizer, buffer, device = setup(args, run_name)
    
    start_update = 1
    global_step = 0
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device) #torch.Size([8, 3, 96, 96])
    next_done = torch.zeros(args.num_envs).to(device) #torch.Size([8])
    num_updates = args.total_timesteps // args.batch_size #9765

    if args.resume is not None:
        global_step, start_update = load_checkpoint(
            args.resume,
            agent,
            optimizer,
            device,
        )

        start_update += 1

    best_return = -np.inf
    for update in range(start_update, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        next_obs, next_done, global_step, avg_ep_ret, avg_ep_len = collect_rollout(
            envs= envs,
            agent= agent,
            buffer= buffer,
            next_obs= next_obs,
            next_done= next_done,
            global_step= global_step,
            args= args,
            device= device,
        )

        if args.track:
            wandb.log({"avg_episodic_return": avg_ep_ret, "avg_episodic_length": avg_ep_len,}, step=global_step,)

        advantages, returns = compute_advantages(agent= agent, buffer= buffer, next_done= next_done, next_obs= next_obs, args= args,)   

        batch = buffer.to_batch(advantages= advantages, returns= returns)
      
        # Optimizing the policy and value network
        metrics= update_agent(agent= agent, optimizer= optimizer, batch= batch, args= args,)

        if update % args.save_frequency == 0:
            checkpoint_path = os.path.join( checkpoint_dir,f"checkpoint_{update}.pt")

            save_checkpoint(
                agent=agent,
                optimizer=optimizer,
                global_step=global_step,
                update=update,
                args=args,
                path=checkpoint_path,
            )

        if avg_ep_ret > best_return:
            best_return = avg_ep_ret

            best_path = os.path.join(checkpoint_dir, "best_model.pt")

            save_checkpoint(
                agent=agent,
                optimizer=optimizer,
                global_step=global_step,
                update=update,
                args=args,
                path=best_path,
            )

        if args.track:
            wandb.log(
                {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy_loss": metrics["entropy_loss"],
                    "total_loss": metrics["total_loss"],
                    "approx_kl": metrics["approx_kl"],
                    "clipfrac": metrics["clipfrac"],
                    "explained_variance": metrics["explained_var"]
                },
                step=global_step,
            )
    checkpoint_path = os.path.join( checkpoint_dir,f"final_model.pt")
    save_checkpoint(
                agent=agent,
                optimizer=optimizer,
                global_step=global_step,
                update=num_updates,
                args=args,
                path=checkpoint_path,
            )

if __name__ == "__main__":
    main()