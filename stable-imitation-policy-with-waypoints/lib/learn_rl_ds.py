#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gym
import zipfile

import torch
import numpy as np

from typing import List
from pathlib import Path
from policy_interface import PlanningPolicyInterface
from utils.log_config import logger

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.policies import serialize
from imitation.data.types import Trajectory
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.bc import BC
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm


class RL_DS(PlanningPolicyInterface):
    """ Learning a policy by looking at state-action pairs demonstrated by an expert.

    The class provides support for Behavioral Cloning and Generative Adverserial
        Imitation Learning, for now.
    """

    def __init__(self, algorithm: str = "GAIL", use_gpu: bool = True,
                 gym_env: str = "taskspace2d", gym_envs_path: str = 'envs',
                 n_envs: int = 16, learner_agent: str = "PPO"):
        """ Initialize a RL_DS object.
        """

        if algorithm not in ["GAIL", "BC"]:
            raise NotImplementedError(f'No support for {algorithm} at the moment')

        self.__device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

        self.__rollouts: List[Trajectory] = []

        self.setup_gym_env(gym_env, gym_envs_path)
        assert n_envs > 2, "Better to have at least two environments"
        self.__vector_envs = DummyVecEnv([lambda: gym.make(f'{gym_env}-v0')] * n_envs)

        self.__learner = self._init_learner(agent=learner_agent)
        self.__algorithm = algorithm

    def fit(self, trajectory: np.ndarray, velocity: np.ndarray, n_epochs_bc: int = 100,
        n_timesteps_gail: int = 3e5, show_stats: bool = True, warm_policy: bool = True,
        n_dems: int = 50):
        """ Fit a GMM and extract parameters for the estimated dynamical systems.

        Args:
            trajectory (np.ndarray): Trajectory data in shape (sample size, dimension).
            velocity (np.ndarray): Velocity data in shape (sample size, dimension).
            tol (float, optional): Tolerance. Defaults to 0.00001.
            show_stats (bool, optional): Whether to show optimization stats.
                Defaults to True.
            warm_policy (bool, optional): Whether to pass a working policy to GAIL.
                Defaults to False.
        """

        self._prepare_dataset(trajectory, velocity, n_dems)

        self.__bc = BC(observation_space=self.__vector_envs.observation_space,
            action_space=self.__vector_envs.action_space, demonstrations=self.__rollouts,
            rng = np.random.default_rng(), policy=self.__learner.policy,
            batch_size=512, ent_weight=0.1, l2_weight=0.1, device=self.__device)

        if self.__algorithm == "BC":
            logger.info(f'Training BC agent on demonstrations for {n_epochs_bc} epochs')
            self.__bc.optimizer = torch.optim.AdamW(self.__learner.policy.parameters())
            self.__bc.train(n_epochs=n_epochs_bc)

        if self.__algorithm == "GAIL":
            if warm_policy:
                self.__bc.optimizer = torch.optim.AdamW(self.__learner.policy.parameters())
                self.__bc.train(n_epochs=100)

            reward_net = BasicRewardNet(self.__vector_envs.observation_space,
                self.__vector_envs.action_space, normalize_input_layer=RunningNorm)

            self.__gail = GAIL(demonstrations=self.__rollouts, demo_batch_size=64,
                gen_replay_buffer_capacity=128, n_disc_updates_per_round=4,
                venv=self.__vector_envs, gen_algo=self.__learner,
                reward_net=reward_net)

            self.__gail.train(n_timesteps_gail)

    def predict(self, trajectory: np.ndarray):
        """ Predict estimated velocities from an input array of states by applying the policy.

        Args:
            trajectory (np.ndarray): Trajectory in shape (sample size, dimension).

        Returns:
            np.ndarray: Estimated velocities in shape (sample size, dimension).
        """

        with torch.no_grad():
            states_tensor = torch.tensor(trajectory, device=self.__device)
            return self.__learner.policy.forward(states_tensor)[0].cpu().numpy()

    def load(self, model_name: str, dir: str = '../../res'):
        """ Load a previously stored model.

        Args:
            model_name (str): Model name.
            dir (str, optional): Path to the load directory. Defaults to '../res'.
        """

        with zipfile.ZipFile(os.path.join(dir, model_name, 'model.zip')) as myzip:
            with myzip.open('policy.pth') as policy_file:
                self.__learner.policy.load_state_dict(torch.load(policy_file))
        logger.info(f'Model {model_name} loaded')

    def save(self, model_name: str, dir: str = '../../res'):
        """ Save the model for later use.

        Args:
            model_name (str): Model name.
            dir (str, optional): Path to the save directory. Defaults to '../res'.
        """

        os.makedirs(dir, exist_ok=True)
        serialize.save_stable_model(Path(f'{dir}/{model_name}'), self.__learner)
        logger.info(f'Model {model_name} saved')

    @staticmethod
    def setup_gym_env(gym_env_name: str, envs_dir: str):
        """Register Gym environment.

        Args:
            gym_env_name (str): Name of the environment.
            envs_dir (str): Path to the envs folder.
        """

        if f'{gym_env_name}.py' not in os.listdir(Path(envs_dir)):
            raise FileNotFoundError(f'No file corresponds to {gym_env_name}.py in envs folder!')

        gym.envs.registration.register(
            id=f'{gym_env_name}-v0',
            entry_point=f'envs.{gym_env_name}:GymExperimentEnv',
            max_episode_steps=500,
        )

    def _init_learner(self, agent: str = "PPO"):
        """ Initialize a learner based on the given agent.

        Args:
            agent (str, optional): Learner agent. Defaults to "PPO".
                Only "PPO" supported for now.
        """

        if agent == "PPO":
            learner = PPO(env=self.__vector_envs, policy=MlpPolicy, batch_size=64,
                          learning_rate=0.01, n_epochs=64, n_steps=300,
                          device=self.__device)
        else:
            raise NotImplementedError("Support for the requested agent not available yet")

        learner.policy = MlpPolicy(observation_space=self.__vector_envs.observation_space,
                                   action_space=self.__vector_envs.action_space,
                                   lr_schedule=RL_DS.linear_schedule(0.01),
                                   net_arch=dict(pi=[128, 128, 128], vf=[64, 64]))

        return learner

    def _prepare_dataset(self, trajs: np.ndarray, vels: np.ndarray, n_dems: int):
        """ Convert npy data to trajectory rollouts.

        Args:
            trajs (np.ndarray): Augmented trajectories.
            vels (np.ndarray): Augmented velocities.
        """

        assert n_dems != 0 and trajs.shape[0] >= 1000, "Expert demonstrations not passed properly"
        len_single_traj = int(trajs.shape[0] / n_dems)

        for traj_idx in range(n_dems):
            # one less action for terminal state
            act = vels[traj_idx * len_single_traj: (traj_idx + 1) * len_single_traj - 1]
            obs = trajs[traj_idx * len_single_traj: (traj_idx + 1) * len_single_traj]

            self.__rollouts.append(Trajectory(obs, act, None, True))

        logger.info(f'Shape of observations: {obs.shape}, \n actions: {act.shape}, '
                    f'number of rolloutes: {len(self.__rollouts)}, '
                    f'and samples per trajectory: {len_single_traj}')

    def linear_schedule(initial_value: float):
        """ Linear learning rate schedule.
        """
        def func(progress_remaining: float) -> float:
            """ Progress will decrease from 1 (beginning) to 0.
            """
            return progress_remaining * initial_value
        return func
