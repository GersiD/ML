import numpy as np
import matplotlib.pyplot as plt
from typing import List
from typing import Callable
from typing import Dict

class FiniteHorizonOptimalStopping:
    def __init__(self, n):
        self.n = n
        self.cur_state = 0
        self.G : List[float] = []
        for _ in range(n + 1):
            reward = np.random.normal(0, 1)
            self.update_gain(reward)

    def reset(self):
        self.cur_state = 0

    def update_gain(self, reward):
        if len(self.G) == 0:
            self.G.append(reward)
        else:
            self.G.append(self.G[-1] + reward)

    def step(self, action):
        if action not in [0, 1]:
            raise ValueError("Action must be 0 or 1")
        reward = 0
        done = False
        if action == 1 or self.cur_state == self.n:
            reward = self.G[self.cur_state]
            done = True
        self.cur_state += 1
        return self.cur_state, reward, done
    
    def simulate(self, policy: Callable):
        """ 
        Simulate the environment with the given policy. 
            Args: policy: A function that takes the environment as input and returns an action [0 keep going, 1 stop]. 
        """
        self.reset()
        reward = 0
        done = False
        while not done:
            action = policy(self)
            state, reward, done = self.step(action)
        return reward

def stop_late(env: FiniteHorizonOptimalStopping):
    return 1 if env.cur_state == env.n else 0

def stop_early(env: FiniteHorizonOptimalStopping):
    return 0 if env.cur_state == env.n else 1

def omniscient(env: FiniteHorizonOptimalStopping):
    return 1 if np.argmax(env.G) == env.cur_state else 0

def random_policy(env: FiniteHorizonOptimalStopping):
    return np.random.choice([0, 1], p=[0.7, 0.3])

def run_simulation(policies: Dict[str, Callable], n: int, env: FiniteHorizonOptimalStopping | None = None):
    """
        Run a simulation with the given policies
        Args: policies: A dictionary of str -> policies to simulate
                n: The number of steps in the simulation
        Returns: A dictionary of str -> rewards for each policy
    """
    env = env or FiniteHorizonOptimalStopping(n)
    rewards = {}
    for policy_name, policy in policies.items():
        rewards[policy_name] = env.simulate(policy)
    return rewards


def main():
    n = 100
    policies = {
        "Stop-Late": stop_late,
        "Stop-Early": stop_early,
        "Random": random_policy,
        "Omniscient": omniscient
    }
    rewards : Dict[str, List[float]] = {}
    for i in range(100):
        temp = run_simulation(policies, n)
        for policy, reward in temp.items():
            if policy not in rewards:
                rewards[policy] = []
            rewards[policy].append(reward)
    # Plot the rewards
    for policy, reward in rewards.items():
        plt.plot(reward, label=policy)
    plt.legend()
    plt.xlabel("Simulation")
    plt.ylabel("Reward")
    plt.title("Optimal Stopping Simulation")
    plt.savefig("optimal_stopping.png")
    plt.show()


if __name__ == '__main__':
    main()
