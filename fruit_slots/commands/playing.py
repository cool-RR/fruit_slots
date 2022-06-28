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


@cli.command()
@click.option('--bananas/--no-bananas', 'produce_bananas', default=True)
@click.option('--lemons/--no-lemons', 'produce_lemons', default=True)
def play_single(*, produce_bananas, produce_lemons):
    from fruit_slots.fruit_slots_env import FruitSlotsEnv
    env = FruitSlotsEnv(produce_bananas=produce_bananas, produce_lemons=produce_lemons)
    model = utils.load_model(produce_bananas=produce_bananas, produce_lemons=produce_lemons)
    print('Starting playing... ')
    observations = env.reset()
    print(env.render())
    print()
    dones = {agent: False for agent in env.agents}
    while False in dones.values():
        actions = {agent: model.predict(observation, deterministic=True)[0]
                   for agent, observation in observations.items()}
        observations, rewards, dones, infos = env.step(actions)
        print(f'Actions: {actions["player_1"]} / {actions["player_2"]}')
        print(env.render())
        print(f'Rewards: {rewards["player_1"]} / {rewards["player_2"]}')
        print()
    print('Done playing.')


