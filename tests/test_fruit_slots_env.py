# Copyright 2022 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import textwrap
import sys
from fruit_slots import FruitSlotsEnv


def test_basics():
    env = FruitSlotsEnv()
    agents = ('player_1', 'player_2')
    for agent in agents:
        observation = env.observe(agent)
        observation_space = env.observation_space(agent)
        assert observation in observation_space

    env.step({'player_1': 3, 'player_2': 4})
    assert env.agent_locations['player_1'] == 3
    assert env.agent_locations['player_2'] == 4
    assert env.render() == textwrap.dedent('''\
        /----------\\
        |   1      |
        |    2     |
        \\----------/''')


def test_long():
    env = FruitSlotsEnv()
    for i in range(1, 501):

        for agent in env.agents:
            observation = env.observe(agent)
            observation_space = env.observation_space(agent)
            assert observation in observation_space

        actions = {agent: 0 for agent in env.agents}
        _, _, _, _ = env.step(actions)
        rendered_env = env.render()
        assert rendered_env.count('1') == 1
        assert rendered_env.count('2') == 1

        if i % 100 == 0:
            assert rendered_env.count('A') == 0
            assert rendered_env.count('a') == 0
            assert rendered_env.count('B') == 1
            assert rendered_env.count('L') == 5
        elif i % 25 == 0:
            assert rendered_env.count('A') == 0
            assert rendered_env.count('a') == 0
            assert rendered_env.count('B') == 1
            assert rendered_env.count('L') == 0
        elif i % 5 == 0:
            assert rendered_env.count('A') == 1
            assert rendered_env.count('a') == 1
            assert rendered_env.count('B') == 0
            assert rendered_env.count('L') == 0
