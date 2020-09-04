import numpy as np
import torch
import random


class EpsilonGreedyExplorationStrategy:
    def __init__(self,
                 epsilon_start: float = 0.01,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 1.0,
                 decay_strategy: str = 'exponential',
                 ensure_exploration: bool = False):
        """
        An Epsilon-Greedy Exploration Strategy

        Args:
            epsilon_start: the initial random action probability
            ensure_exploration: whether to guarantee that random actions will never be the greedy the action.
        """

        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.epsilon = epsilon_start

        self.ensure_exploration: ensure_exploration


    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        self._epsilon = value


    def choose_action(self, model, state, device: str):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        model.eval()

        with torch.no_grad():
            action_values = model(state)
            q_values = action_values.cpu().data.numpy().squeeze()

        model.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(q_values)
        else:
            ## TODO: implement  ensure_exploration
            return random.choice(np.arange(len(q_values)))


    def step(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
