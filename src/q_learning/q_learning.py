import gym

from q_learning.performance_analyzer import PerformanceAnalyzer
from q_learning.q_table import QTable


class QLearning:
    """
    Summary:
    QLearning learns how to maneuver a car in order to reach a goal at the top of a hill using a Q-Table. The simulator
    provides position and velocity which the Q-table uses in order to make an action. Actions are move Left, do Nothing,
    or move Right. It will render the simulation at a specified interval. Once the maximum number of episodes have been
    reached it will stop learning and graph the performance metrics of the actor.
    """

    _EPISODES = 5000
    _RENDER_PERIOD = 500

    def __init__(self):
        self._env = gym.make("MountainCar-v0")  # Environment
        self._env.reset()

        max_observation_values = self._env.observation_space.high
        min_observation_values = self._env.observation_space.low
        num_of_actions = self._env.action_space.n
        self._q_table = QTable(max_observation_values, min_observation_values, num_of_actions)

        self._performance_analyzer = PerformanceAnalyzer()

    def run(self):
        for episode in range(self._EPISODES):
            render = self._to_render(episode)
            state = self._env.reset()

            done = False
            episode_reward = 0
            while not done:  # Episode will terminate after a certain amount of time has elapsed
                action = self._q_table.get_action(state)

                # new_state are things sensed by environment, position and velocity
                # reward is -1 until the car reaches the flag, then it becomes 0
                new_state, reward, done, _ = self._env.step(action)

                if not done:
                    # Since every frame is -1 reward, it will naturally try to achieve goal in shortest amount of time
                    self._q_table.update(new_state, reward, state, action)
                elif new_state[0] >= self._env.goal_position:  # new_state[0] is car position
                    self._q_table.goal_achieved(state, action)
                    print(f"Goal achieved on episode {episode}")

                state = new_state
                episode_reward += reward
                if render:
                    self._env.render()

            self._performance_analyzer.add_reward(episode_reward)

        self._performance_analyzer.show_metrics()

    def _to_render(self, episode):
        if episode % self._RENDER_PERIOD == 0:
            print(f"Rendering episode: {episode}")
            return True
        else:
            return False

    def __del__(self):
        if self._env is not None:
            self._env.close()
