import random
import sys
import itertools
import warnings
import functools
import warnings
warnings.filterwarnings('ignore')

import unittest.mock
import pathlib

import click
import numpy as np
import gym.vector.utils
import supersuit as ss
from stable_baselines3 import PPO
import stable_baselines3.ppo
import pettingzoo.utils.conversions
from pettingzoo.classic import chess_v5


log_path: pathlib.Path = pathlib.Path.home() / '.fruit_slots' / 'logs'


class TensorboardCallback(stable_baselines3.common.callbacks.BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def _on_step(self):
        from stable_baselines3.common.utils import safe_mean

        log_interval = self.locals['log_interval']
        iteration = self.locals['iteration']
        model = self.locals['self']
        ep_info_buffer = model.ep_info_buffer
        if (log_interval is not None and iteration % log_interval == 0 and
            len(ep_info_buffer) > 0 and len(ep_info_buffer[0]) > 0):

            for metric_name in ep_info_buffer[0].get('loggable_metrics', ()):
                self.logger.record(
                    f'rollout/mean_{metric_name}',
                    safe_mean([ep_info[metric_name] for ep_info in ep_info_buffer])
                )


        return True

@click.group()
def cli():
    pass


@cli.command()
def run():
    from .fruit_slots_env import FruitSlotsEnv
    load_existing = ('-l' in sys.argv)
    env = original_env = FruitSlotsEnv(produce_lemons=True)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    if '-p' in sys.argv:
        env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')
    else:
        env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    env = stable_baselines3.common.vec_env.VecMonitor(
        env,
        info_keywords=(original_env.custom_metrics + ('loggable_metrics',))
    )
    model = stable_baselines3.PPO(stable_baselines3.ppo.MlpPolicy, env, n_steps=32,
                                  tensorboard_log=log_path, verbose=True)
    if load_existing:
        model = PPO.load('policy', env=env)
        print('Loaded policy from file.')
    else:
        print('Starting learning... ')
        model.learn(total_timesteps=1_000_000, n_eval_episodes=30, eval_freq=1,
                    callback=TensorboardCallback())
        model.save('policy')
        print('Done learning.')

    print('Starting playing... ')
    observations = original_env.reset()
    print(original_env.render())
    print()
    dones = {agent: False for agent in original_env.agents}
    while False in dones.values():
        actions = {agent: model.predict(observation, deterministic=True)[0]
                   for agent, observation in observations.items()}
        observations, rewards, dones, infos = original_env.step(actions)
        print(actions)
        print(original_env.render())
        print(rewards)
        print()
        # original_env.render()
    print('Done playing.')


