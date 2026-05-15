import gymnasium as gym
import numpy as np

class ClipRewardCarRacing(gym.RewardWrapper):
    def __init__(self, env, max_reward=1.0):
        super().__init__(env)
        self.max_reward = max_reward

    def reward(self, reward):
        # Mantiene inalterate le penalità (valori negativi), ma taglia i picchi positivi
        return np.clip(reward, a_min=None, a_max=self.max_reward)
    
def make_env(gym_id, seed, idx, render_mode, capture_video, clip_reward, run_name,):
    def thunk():
        env = gym.make(gym_id, continuous=False, render_mode=render_mode)
        if clip_reward is not None:
            env = ClipRewardCarRacing(env,max_reward= clip_reward)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        new_obs_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, env.observation_space.shape[1], env.observation_space.shape[2]),
            dtype=np.uint8
        )
        env = gym.wrappers.TransformObservation(env, lambda obs: obs[:84, :, :], observation_space=new_obs_space)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk

def make_vec_envs(num_envs, gym_id, seed, render_mode, capture_video, clip_reward, run_name):
    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id, seed + i,i, render_mode, capture_video, clip_reward, run_name) for i in range(num_envs)]
    )
    return envs