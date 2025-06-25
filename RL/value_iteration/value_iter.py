import numpy as np

# Define a small MDP 3 states, 3 actions
# | s0 | s1 | s2 |
STATES = [0, 1, 2]

# Actions are a0 (go left), a1 (go right)
class Action:
    LEFT = 0
    RIGHT = 1
    @staticmethod
    def __call__(item):
        match item:
            case 0:
                return "LEFT"
            case 1:
                return "RIGHT"
        return None
ACTIONS = [Action.LEFT, Action.RIGHT]

# Transition probabilities
S0_TRANSITIONS = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])
S1_TRANSITIONS = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0]])
S2_TRANSITIONS = np.array([[0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
TRANSITIONS = np.array([S0_TRANSITIONS, S1_TRANSITIONS, S2_TRANSITIONS]) # shape (S, A, S)

# Rewards
REWARDS = np.array([[0.0, 0.0], # no rewards for s0
                    [0.0, 1.0],  # reward for moving to s2 from s1
                    [0.0, 1.0]   # reward for staying in s2
            ])  # S X A matrix, where each state has a reward for each action

# Discount factor
GAMMA = 0.9

def value_iteration(transitions, rewards, gamma, theta=1e-6, max_iterations=1000):
    num_states = transitions.shape[0]

    Q_values = np.zeros((num_states, len(ACTIONS)))
    itr = 0 
    while itr < max_iterations:  # Limit iterations to prevent infinite loop
        Q_values_prev = Q_values.copy()
        # SXA + scaler * SXAXS @ S = SXA + scaler * SXA
        Q_values = rewards + gamma * transitions @ np.max(Q_values_prev, axis=1)
        # Check for convergence
        if np.max(np.abs(Q_values - Q_values_prev)) < theta:
            return Q_values
        itr += 1
    print(f"Value iteration stopped after {max_iterations} iterations without convergence.")
    return Q_values

def main():
    q_values = value_iteration(TRANSITIONS, REWARDS, GAMMA)
    print("Q-values after value iteration:")
    print(q_values)
    print("Optimal policy:")
    optimal_policy = np.argmax(q_values, axis=1)
    print("Optimal policy in state 0:", Action()(optimal_policy[0]))
    print("Optimal policy in state 1:", Action()(optimal_policy[1]))
    print("Optimal policy in state 2:", Action()(optimal_policy[2]))

if __name__ == "__main__":
    main()
