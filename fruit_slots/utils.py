# Copyright 2022 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import os
from typing import Optional
import pathlib

if False:
    # Used only for typing.
    import stable_baselines3


fruit_slots_home_path = pathlib.Path(os.environ.get('FRUIT_SLOTS_HOME_PATH',
                                                    pathlib.Path.home() / '.fruit_slots'))
log_path: pathlib.Path = fruit_slots_home_path / 'logs'
model_path: pathlib.Path = fruit_slots_home_path / 'models'


def prevent_tensorflow_spam() -> None:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def save_model(model: stable_baselines3.PPO, *,
               i_agent: Optional[int] = None, produce_bananas: bool = True,
               produce_lemons: bool = True) -> None:
    agent_path = make_agent_path(i_agent=i_agent,
                                 produce_bananas=produce_bananas, produce_lemons=produce_lemons)
    print(f'Writing model to {agent_path}')
    model.save(agent_path)


def make_agent_path(*, i_agent: Optional[int] = None, produce_bananas: bool = True,
                    produce_lemons: bool = True) -> pathlib.Path:
    result = 'agent-'
    if i_agent is None:
        result += 'single-'
    else:
        result += f'{i_agent}-'

    result += 'a'
    if produce_bananas: result += 'b'
    if produce_lemons: result += 'l'

    result += '.zip'

    return model_path / result


def load_model(*, i_agent: Optional[int] = None, produce_bananas: bool = True,
               produce_lemons: bool = True) -> stable_baselines3.PPO:
    from .fruit_slots_env import FruitSlotsEnv
    import stable_baselines3
    agent_path = make_agent_path(i_agent=i_agent,
                                 produce_bananas=produce_bananas, produce_lemons=produce_lemons)
    if not agent_path.exists():
        raise Exception('You should train before you can use the agents.') from \
                                                                       FileNotFoundError(agent_path)
    print(f'Reading model from {agent_path}')
    return stable_baselines3.PPO.load(
        agent_path,
        env=FruitSlotsEnv.make_and_wrap(is_parallel=False,
                                        produce_bananas=produce_bananas,
                                        produce_lemons=produce_lemons)
    )



