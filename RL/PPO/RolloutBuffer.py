class RolloutBuffer:
    """
    RolloutBuffer class, which is used to store the experiences of the agent during training.
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def insert(self, state, action, logprob, state_value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.state_values.append(state_value)
