import gym
from q_learning.actions import Actions


class QLearning:
    def __init__(self):
        self._environment = gym.make("MountainCar-v0")
        self._environment.reset()

        self._done = False

    def run(self):
        while not self._done:
            # new_state are things sensed by environment, position and velocity
            new_state, reward, done, _ = self._environment.step(Actions.Right.value)

            self._environment.render()
