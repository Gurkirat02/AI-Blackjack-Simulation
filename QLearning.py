import random

class QLearning():

    def __init__(self, alpha = 0.1, gamma = 0.9, epsilon = 0.1):
        # Initialize Q-values for state-action pairs
        # {key = (state, action): value = q-value}
        self.q_values = {}

        self.actions = ['HIT', 'STAND']
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation tradeoff

    # Function to get the Q-value for a state-action pair, returns 0 if the key is not present in the dictionary
    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0)
    
    # Function to update Q-values based on the Bellman equation
    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        max_next_q = max(self.get_q_value(next_state, a) for a in self.actions)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_values[(state, action)] = new_q
        # print(self.q_values)

    # Function to choose an action using epsilon-greedy strategy
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return random.choice(self.actions)  
        else:
            # Exploit: choose the action with the maximum q-value
            return max(self.actions, key=lambda a: self.get_q_value(state, a))  
