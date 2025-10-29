import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.01, gamma=0.9, epsilon=1.0, eps_decay=0.995, eps_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.Q = {}

    def get_state_key(self, state):
        return tuple(np.round(state, 2))

    def get_action(self, state, explore=True):
        key = self.get_state_key(state)
        if np.random.rand() < self.epsilon and explore:
            return np.random.randint(self.action_size)
        return np.argmax(self.Q.get(key, np.zeros(self.action_size)))

    def learn(self, state, action, reward, next_state, done):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        q_values = self.Q.get(key, np.zeros(self.action_size))
        next_q = self.Q.get(next_key, np.zeros(self.action_size))
        target = reward + self.gamma * np.max(next_q) * (1 - done)
        q_values[action] += self.alpha * (target - q_values[action])
        self.Q[key] = q_values
        if done:
            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
