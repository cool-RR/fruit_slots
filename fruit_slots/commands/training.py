# Copyright 2022 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import random
import sys
import itertools
import warnings
import functools

import unittest.mock
import pathlib

import click
import numpy as np

from fruit_slots import utils
from . import cli

if False:
    # Used only for typing.
    import stable_baselines3


@cli.command()
@click.option('-p', '--parallel', 'is_parallel', default=False)
@click.option('--bananas/--no-bananas', 'produce_bananas', default=True)
@click.option('--lemons/--no-lemons', 'produce_lemons', default=True)
@click.option('-t', '--total-timesteps', default=1_000_000)
@click.option('-v', '--verbose', default=False, is_flag=True)
def train_single(*, is_parallel, produce_bananas, produce_lemons, total_timesteps, verbose):
    import stable_baselines3
    from fruit_slots.fruit_slots_env import FruitSlotsEnv
    utils.prevent_tensorflow_spam()

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

    env = FruitSlotsEnv.make_and_wrap(is_parallel=is_parallel,
                                      produce_bananas=produce_bananas,
                                      produce_lemons=produce_lemons)
    model = stable_baselines3.PPO(stable_baselines3.ppo.MlpPolicy, env, n_steps=32,
                                  tensorboard_log=utils.log_path, verbose=verbose)

    print('Starting learning... ')
    model.learn(total_timesteps=total_timesteps, n_eval_episodes=30, eval_freq=1,
                callback=TensorboardCallback())
    print('Done learning.')
    utils.save_model(model, produce_bananas=produce_bananas, produce_lemons=produce_lemons)
