import torch
import numpy as np
import gym_minigrid  # Needed for Utils.make_env
from gym_minigrid.wrappers import *

import gym
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.experiments.tasks.preprocessor_base import PreprocessorBase
from continual_rl.utils.utils import Utils
from continual_rl.utils.env_wrappers import FrameStack, LazyFrames


class MiniGridToPyTorch(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space['image'].shape

        # Minigrid tasks are represented by integers in the range [0, 10]
        # Specifically, each of the 3 channels is [OBJECT_IDX, COLOR_IDX, STATE]
        # OBJECT_IDX is [0, 10], COLOR_IDX is [0, 5], and STATE is [0, 2]
        # (https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=10,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        processed_observation = torch.tensor(observation['image'])
        processed_observation = processed_observation.permute(2, 0, 1)
        return processed_observation


class MiniGridPreprocessor(PreprocessorBase):
    def __init__(self, env_spec, time_batch_size):
        self.env_spec = self._wrap_env(env_spec, time_batch_size)
        dummy_env, _ = Utils.make_env(self.env_spec)
        super().__init__(dummy_env.observation_space)

    def _wrap_env(self, env_spec, time_batch_size):
        frame_stacked_env_spec = lambda: FrameStack(MiniGridToPyTorch(Utils.make_env(env_spec)[0]), time_batch_size)
        return frame_stacked_env_spec

    def preprocess(self, batched_obs):
        assert isinstance(batched_obs[0], LazyFrames), f"Observation was of unexpected type: {type(batched_obs[0])}"
        # Minigrid images are [H, W, C], so rearrange to pytorch's expectations.
        return torch.stack([obs.to_tensor() for obs in batched_obs])

    def render_episode(self, episode_observations):
        """
        Turn a list of observations gathered from the episode into a video that can be saved off to view behavior.
        """
        # Note: the 3 channels aren't really representing RGB, so this is a convenient but not necessarily
        # optimally understandable representation
        return torch.stack(episode_observations).unsqueeze(0).float() / self.observation_space.high.max()


class MiniGridTask(TaskBase):
    """
    MiniGrid has a custom observation format, so we have a separate Task type to handle parsing it
    """
    def __init__(self, task_id, action_space_id, env_spec, num_timesteps, time_batch_size, eval_mode):
        preprocessor = MiniGridPreprocessor(env_spec, time_batch_size)
        dummy_env, _ = Utils.make_env(preprocessor.env_spec)
        action_space = dummy_env.action_space

        super().__init__(task_id, action_space_id, preprocessor, preprocessor.env_spec, preprocessor.observation_space,
                         action_space, num_timesteps, eval_mode)

def make_minigrid(
    env_name,
    env_seed=42,
):
    env = gym.make(env_name)
    #env = RGBImgPartialObsWrapper(env)  # Get rid of the 'mission' field
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    #env = ImgObsWrapper(env)
    #print("Creating MiniGrid env with seed {}".format(env_seed))
    #env = ReseedWrapper(env, [env_seed])
    return env


def get_single_minigrid_task(task_id, action_space_id, env_name, num_timesteps, eval_mode=False, env_seed=42):
    return MiniGridTask(
        task_id=task_id,
        action_space_id=action_space_id,
        env_spec=lambda: make_minigrid(env_name, env_seed=env_seed),
        num_timesteps=num_timesteps,
        time_batch_size=1,  # no framestack
        eval_mode=eval_mode,
    )