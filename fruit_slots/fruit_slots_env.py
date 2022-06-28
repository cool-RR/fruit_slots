# Copyright 2022 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import random

import numpy as np
import gym.vector.utils
import pettingzoo


N_SLOTS = 10

REWARD_NOTHING = -0.1
REWARD_APPLE = 1
REWARD_BANANA = 5
REWARD_LEMON = -20

EPISODE_LENGTH = 500


class FruitSlotsEnv(pettingzoo.ParallelEnv):

    metadata = {
        "render_modes": ["human"],
        "name": "fruit_slots",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    @staticmethod
    def make_and_wrap(*, produce_bananas=True, produce_lemons=True, is_parallel=False):
        import supersuit as ss
        import stable_baselines3

        env = original_env = FruitSlotsEnv(
            produce_bananas=produce_bananas, produce_lemons=produce_lemons
        )
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        if is_parallel:
            env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')
        else:
            env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
        env = stable_baselines3.common.vec_env.VecMonitor(
            env,
            info_keywords=(original_env.custom_metrics + ('loggable_metrics',))
        )
        env.original_env = env
        return env


    def __init__(self, *, produce_bananas=True, produce_lemons=True):
        if produce_lemons:
            assert produce_bananas
        self.produce_bananas = produce_bananas
        self.produce_lemons = produce_lemons
        self.possible_agents = ['player_1', 'player_2']
        self._action_space = gym.spaces.Discrete(N_SLOTS)
        self._observation_space = gym.spaces.Box(low=0, high=1, shape=(2, N_SLOTS, 6),
                                                 dtype=bool)
        self.action_spaces = {name: self._action_space for name in self.possible_agents}
        self.observation_spaces = {name: self._observation_space for name in self.possible_agents}
        self.custom_metrics = (
            'cumulative_reward',
            'cumulative_visible_apple_reward',
            'cumulative_invisible_apple_reward',
            *(('cumulative_banana_reward',) if produce_bananas else ()),
            *(('cumulative_lemon_reward',) if produce_lemons else ()),
        )

        self.reset()


    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None):
        self.agents = self.possible_agents.copy()

        self._cumulative_rewards = {name: 0 for name in self.agents}
        self._cumulative_visible_apple_rewards = {name: 0 for name in self.agents}
        self._cumulative_invisible_apple_rewards = {name: 0 for name in self.agents}
        self._cumulative_banana_rewards = {name: 0 for name in self.agents}
        self._cumulative_lemon_rewards = {name: 0 for name in self.agents}

        self.agent_locations = {agent: random.randint(0, N_SLOTS - 1) for agent in self.agents}
        self.i_step = 0
        self._remove_all_fruits()
        return self.get_observations()

    def _remove_all_fruits(self):
        self.apple_locations = (set(), set())
        self.visible_apple_locations = (
            (set(), set()),
            (set(), set()),
        )
        self.banana_locations = (set(), set())
        self.lemon_locations = (set(), set())


    def observe(self, agent):
        observation = self._observation_space.low.copy()
        current_agent = agent
        i_current_agent = self.agents.index(current_agent)
        i_other_agent = 1 - i_current_agent

        CHANNEL_STATIC_FALSE = 0
        CHANNEL_STATIC_TRUE = 1
        CHANNEL_AGENT_LOCATIONS = 2
        CHANNEL_APPLE_LOCATIONS = 3
        CHANNEL_BANANA_LOCATIONS = 4
        CHANNEL_LEMON_LOCATIONS = 5

        # Static channels for voodoo reasons:
        observation[:, :, CHANNEL_STATIC_FALSE] = False
        observation[:, :, CHANNEL_STATIC_TRUE] = True

        # Locations of both agents:
        for agent, location in self.agent_locations.items():
            i_agent = self.agents.index(agent)
            observation[i_agent, location, CHANNEL_AGENT_LOCATIONS] = True

        if i_current_agent == 1:
            # Mirror image of agents' locations:
            observation[:, :, CHANNEL_AGENT_LOCATIONS] = \
                                                       observation[::-1, :, CHANNEL_AGENT_LOCATIONS]

        visible_apple_locations_for_current_agent = self.visible_apple_locations[i_current_agent]
        ordered_visible_apple_locations = (visible_apple_locations_for_current_agent if i_agent == 0
                                           else visible_apple_locations_for_current_agent[::-1])
        for i, visible_apple_locations in enumerate(ordered_visible_apple_locations):
            for visible_apple_location in visible_apple_locations:
                observation[i, visible_apple_location, CHANNEL_APPLE_LOCATIONS] = True

        if self.banana_locations[i_other_agent]:
            (banana_location,) = self.banana_locations[i_other_agent]
            observation[1, banana_location, CHANNEL_BANANA_LOCATIONS] = True

        for lemon_location in self.lemon_locations[i_other_agent]:
            observation[1, lemon_location, CHANNEL_LEMON_LOCATIONS] = True

        assert observation.shape == self._observation_space.shape
        return observation


    def get_observations(self):
        return {agent: self.observe(agent) for agent in self.agents}


    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        dones = {agent: self.i_step >= 500 for agent in self.agents}
        possible_new_fruit_locations = tuple(set(range(N_SLOTS)) - set(self.agent_locations)
                                             - set(actions.values()))
        self.agent_locations.update(actions)

        for i_current_agent, (current_agent, action) in enumerate(actions.items()):
            i_other_agent = 1 - i_current_agent
            other_agent = self.agents[i_other_agent]

            ### Dealing with agent eating apple: ###################################################
            #                                                                                      #
            apple_locations_for_current_agent = self.apple_locations[i_current_agent]
            try:
                apple_locations_for_current_agent.remove(action)
            except KeyError:
                pass
            else:
                # Agent ate an apple.
                rewards[current_agent] += REWARD_APPLE

                try:
                    self.visible_apple_locations[i_current_agent][i_current_agent].remove(action)
                except KeyError:
                    # Agent ate an invisible apple.
                    self._cumulative_invisible_apple_rewards[current_agent] += REWARD_APPLE
                    self.visible_apple_locations[i_other_agent][i_current_agent].remove(action)
                else:
                    # Agent ate a visible apple.
                    assert (action not in
                            self.visible_apple_locations[i_other_agent][i_current_agent])
                    self._cumulative_visible_apple_rewards[current_agent] += REWARD_APPLE
            #                                                                                      #
            ### Finished dealing with agent eating apple. ##########################################

            ### Dealing with agent eating banana: ##################################################
            #                                                                                      #
            try:
                self.banana_locations[i_current_agent].remove(action)
            except KeyError:
                pass
            else:
                # Agent ate a banana.
                rewards[current_agent] += REWARD_BANANA
                rewards[other_agent] += REWARD_BANANA
                self._cumulative_banana_rewards[current_agent] += REWARD_BANANA
                self._cumulative_banana_rewards[other_agent] += REWARD_BANANA
            #                                                                                      #
            ### Finished dealing with agent eating banana. #########################################


            ### Dealing with agent eating lemon: ###################################################
            #                                                                                      #

            try:
                self.lemon_locations[i_current_agent].remove(action)
            except KeyError:
                pass
            else:
                # Agent ate a lemon.
                rewards[current_agent] = REWARD_NOTHING
                rewards[other_agent] = REWARD_LEMON
                self._cumulative_lemon_rewards[other_agent] += REWARD_LEMON

            #                                                                                      #
            ### Finished dealing with agent eating lemon. ##########################################

            if rewards[current_agent] == 0:
                rewards[current_agent] = REWARD_NOTHING

            self._cumulative_rewards[current_agent] += rewards[current_agent]

            infos[current_agent] = {
                metric: getattr(self, f'_{metric}s')[current_agent]
                for metric in self.custom_metrics
            }
            infos[current_agent]['loggable_metrics'] = self.custom_metrics

            self.agent_locations[current_agent] = action

        ### Advancing turn and dealing with scheduled events: ######################################
        #                                                                                          #
        self.i_step += 1

        if self.i_step % 5 == 0:
            self._remove_all_fruits()

            if self.produce_lemons and self.i_step % 100 == 0:
                i_agent_on_banana_side, i_agent_on_lemon_side = random.sample(range(2), 2)
                new_banana_location, *new_lemon_locations = \
                                                      random.sample(possible_new_fruit_locations, 6)
                self.banana_locations[i_agent_on_banana_side].add(new_banana_location)
                self.lemon_locations[i_agent_on_lemon_side].update(new_lemon_locations)

            elif self.produce_bananas and self.i_step % 25 == 0:
                new_banana_location = random.choice(possible_new_fruit_locations)
                random.choice(self.banana_locations).add(new_banana_location)

            elif self.i_step % 5 == 0:
                new_apple_location = random.choice(possible_new_fruit_locations)
                for apple_locations in self.apple_locations:
                    apple_locations.add(new_apple_location)
                i_agent_that_can_see_new_apple_pair = random.randint(0, 1)
                for i_agent in range(2):
                    self.visible_apple_locations[i_agent_that_can_see_new_apple_pair][i_agent]. \
                                                                             add(new_apple_location)
        #                                                                                          #
        ### Finished advancing turn and dealing with scheduled events. #############################

        return self.get_observations(), rewards, dones, infos


    def render(self, mode='human'):
        result = [([' '] * N_SLOTS) for i_agent in range(2)]
        for agent, agent_location in self.agent_locations.items():
            i_agent = self.agents.index(agent)
            result[i_agent][agent_location] = str(i_agent + 1)
        for i_agent, apple_locations in enumerate(self.apple_locations):
            for apple_location in apple_locations:
                apple_pair_is_visible = (apple_location in
                                         self.visible_apple_locations[i_agent][i_agent])
                result[i_agent][apple_location] = ('A' if apple_pair_is_visible else 'a')
        for i_agent, banana_locations in enumerate(self.banana_locations):
            if banana_locations:
                (banana_location,) = banana_locations
                result[i_agent][banana_location] = 'B'

        for i_agent, lemon_locations in enumerate(self.lemon_locations):
            for lemon_location in lemon_locations:
                result[i_agent][lemon_location] = 'L'

        top_horizontal_line = '/' + '-' * N_SLOTS + '\\'
        bottom_horizontal_line = '\\' + '-' * N_SLOTS + '/'
        return (top_horizontal_line + '\n|' + ''.join(result[0]) + '|\n|' + ''.join(result[1]) +
                '|\n' + bottom_horizontal_line)


    def close(self):
        pass
