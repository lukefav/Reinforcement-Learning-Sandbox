import gym
import time

from q_learning.actions import Actions
from q_learning.q_table import QTable


class QLearning:

    _EPISODES = 25000
    _RENDER_PERIOD = 2000

    def __init__(self):
        self._env = gym.make("MountainCar-v0")  # Environment
        self._env.reset()

        max_observation_values = self._env.observation_space.high
        min_observation_values = self._env.observation_space.low
        num_of_actions = self._env.action_space.n
        self._q_table = QTable(max_observation_values, min_observation_values, num_of_actions)

    def run(self):
        for episode in range(self._EPISODES):
            render = self._render(episode)
            state = self._env.reset()

            done = False
            frames = 0
            while not done:
                action = self._q_table.get_action(state)

                # new_state are things sensed by environment, position and velocity
                # reward is -1 until the car reaches the flag, then it becomes 0
                new_state, reward, done, _ = self._env.step(action)

                if not done:
                    # Since every frame is -1 reward, it will naturally try to achieve goal in shortest amount of time
                    self._q_table.update(new_state, reward, state, action)
                elif new_state[0] >= self._env.goal_position:  # new_state[0] is car position
                    self._q_table.goal_achieved(state, action)
                    print(f"Goal achieved on episode {episode} in {frames} frames.")

                state = new_state
                frames += 1
                if render:
                    self._env.render()

    def _render(self, episode):
        if episode % self._RENDER_PERIOD == 0:
            print(f"Rendering episode: {episode}")
            return True
        else:
            return False

    def __del__(self):
        if self._env is not None:
            self._env.close()


"""
Notes:
    - Q-table, for every position of state and velocity we look up the corresponding Q-Value
"""