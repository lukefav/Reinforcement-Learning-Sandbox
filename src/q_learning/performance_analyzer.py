import matplotlib.pyplot as plt


class PerformanceAnalyzer:

    _BLOCK_SIZE = 100

    def __init__(self):
        self._episode_rewards = []
        self._reward_metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}

        self._temp_episode_count = 0
        self._total_episode_count = 0

    def add_reward(self, reward):
        self._episode_rewards.append(reward)
        self._temp_episode_count += 1
        self._total_episode_count += 1

        if self._temp_episode_count < self._BLOCK_SIZE:
            return

        self._populate_reward_metrics()

    def _populate_reward_metrics(self):
        episode_num = self._total_episode_count
        avg_reward = sum(self._episode_rewards)/len(self._episode_rewards)
        min_reward = min(self._episode_rewards)
        max_reward = max(self._episode_rewards)

        self._reward_metrics['ep'].append(episode_num)
        self._reward_metrics['avg'].append(avg_reward)
        self._reward_metrics['min'].append(min_reward)
        self._reward_metrics['max'].append(max_reward)
        print(f"Episode: {episode_num} avg: {'%.3f' % avg_reward} min: {min_reward} max:{max_reward}")

        self._temp_episode_count = 0
        self._episode_rewards.clear()

    def show_metrics(self):
        plt.plot(self._reward_metrics['ep'], self._reward_metrics['avg'], label='avg')
        plt.plot(self._reward_metrics['ep'], self._reward_metrics['min'], label='min')
        plt.plot(self._reward_metrics['ep'], self._reward_metrics['max'], label='max')
        plt.legend(loc=0)
        plt.show()
