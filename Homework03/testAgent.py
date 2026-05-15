import torch
import numpy as np
import argparse
import os
from agent import Agent
from car_race import make_vec_envs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-folder", type=str, default=None,
        help="the folder in which to save the videos")
    parser.add_argument("--gym-id", type=str, default="CarRacing-v3",
        help="the id of the gym environment")
    parser.add_argument("--seed", type=int, default=1234,
        help="seed of the experiment")
    parser.add_argument("--capture-video", action=argparse.BooleanOptionalAction, default=False,
        help="weather to capture videos of the agent performances")
    parser.add_argument("--model-path", type=str, default=None,
        help="path to the weigths of the model")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=10,
        help="the number of parallel game environments")
    parser.add_argument("--num-episodes", type=int, default=100,
        help="the number of episodes to run")
    parser.add_argument("--render-mode", type=str, default=None, choices=["human", "rgb_array", None],
        help="gymnasium render mode")
    
    args = parser.parse_args()
    if args.render_mode is None:
        args.capture_video = False

    return args

def main():
    args = parse_args()

    envs = make_vec_envs(
        num_envs= args.num_envs,
        gym_id= args.gym_id, 
        seed= args.seed,
        render_mode= args.render_mode,
        capture_video= args.capture_video,
        clip_reward=None,
        run_name= args.video_folder,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(envs).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    episode_returns = []

    next_obs, _ = envs.reset()
    while len(episode_returns) < args.num_episodes:

        next_obs = torch.Tensor(next_obs).to(device)
        action, logprob, _, value = agent.get_value_and_action(next_obs)
        next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())

        if "episode" in info:
            done_envs = info["_episode"]
            returns = info["episode"]["r"]

            for i, done in enumerate(done_envs):
                if done:
                    if len(episode_returns) >= args.num_episodes:
                        break
            
                    episode_returns.append(returns[i])
                    print(f"Completed episodes: "f"{len(episode_returns)}/{args.num_episodes} ")

    envs.close()
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    min_return = np.min(episode_returns)
    max_return = np.max(episode_returns)
    success_rate = np.mean(np.array(episode_returns) > 900)

    print(f"Mean return over {args.num_episodes} episodes: {mean_return:.2f}")
    print(f"Std return: {std_return:.2f}")
    print(f"Min return: {min_return:.2f}")
    print(f"Max return: {max_return:.2f}")
    print(f"Success rate (>900): {success_rate * 100:.2f}%")

    results_dir = "agents_performances"
    os.makedirs(results_dir, exist_ok=True)
    model_path = args.model_path

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    exp_name = os.path.basename(os.path.dirname(model_path))

    results_file = os.path.join(results_dir,f"{exp_name}_{model_name}.txt")
    
    # Salvataggio metriche
    with open(results_file, "w") as f:
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Environment: {args.gym_id}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Number of episodes: {args.num_episodes}\n")
        f.write(f"Number of envs: {args.num_envs}\n\n")

        f.write(f"Mean return: {mean_return:.2f}\n")
        f.write(f"Std return: {std_return:.2f}\n")
        f.write(f"Min return: {min_return:.2f}\n")
        f.write(f"Max return: {max_return:.2f}\n")
        f.write(f"Success rate (>900): {success_rate * 100:.2f}%\n")

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()