import numpy as np




class QTable:

    _AMOUNT_DISCRETE_STATES = 20
    _LEARNING_RATE = 0.1
    _DISCOUNT = 0.95  # Measure of how important we find future rewards over current rewards

    def __init__(self, max_observation_values, min_observation_values, num_of_actions):
        # os is Observation Space
        self._os_space_max = max_observation_values
        self._os_space_min = min_observation_values

        discrete_os_steps = [self._AMOUNT_DISCRETE_STATES]*len(max_observation_values)
        self._discrete_os_step_values = (self._os_space_max - self._os_space_min)/discrete_os_steps

        # Generates a random Q-Value per action at discrete steps through the observation space
        self._q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_steps + [num_of_actions]))

    def get_action(self, current_state):
        discrete_state = self._get_discrete_state(current_state)
        q_values = self._q_table[discrete_state]
        action_num = np.argmax(q_values)

        # 0 is move left, 1 is do nothing, 2 is move right
        return action_num

    def update(self, new_state, new_reward, current_state, current_action):
        new_discrete_state = self._get_discrete_state(new_state)
        current_discrete_state = self._get_discrete_state(current_state)

        max_future_q = np.max(self._q_table[new_discrete_state])
        current_q = self._q_table[current_discrete_state + (current_action,)]

        new_q = self._calculate_new_q_value(current_q, new_reward, max_future_q)
        self._q_table[current_discrete_state + (current_action,)] = new_q

    def goal_achieved(self, current_state, current_action):
        discrete_state = self._get_discrete_state(current_state)
        self._q_table[discrete_state + (current_action,)] = 0  # Reward for reaching goal is 0

    def _get_discrete_state(self, continuous_state):
        discrete_state = (continuous_state - self._os_space_min)/self._discrete_os_step_values
        return tuple(discrete_state.astype(np.int))

    def _calculate_new_q_value(self, current_q, reward, max_future_q):
        return (1 - self._LEARNING_RATE)*current_q + self._LEARNING_RATE*(reward + self._DISCOUNT*max_future_q)
