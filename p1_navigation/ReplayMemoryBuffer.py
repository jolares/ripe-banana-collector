import random
from collections import deque, namedtuple
import numpy as np
import torch


## Default Values

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
MIN_NUM_BATCHES = 5


class ReplayMemoryBuffer:
    def __init__(self,
                 buffer_size: int = BUFFER_SIZE,
                 batch_size: int = BATCH_SIZE,
                 device: str = DEVICE,
                 seed: int = 0):

        self.device = device
        self.memory_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)


    def __len__(self):
        return len(self.memory_buffer)


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience_sample = self.experience(state, action, reward, next_state, done)
        self.memory_buffer.append(experience_sample)


    def sample(self, batch_size: int = None, device: str = None):

        if not batch_size:
            batch_size = self.batch_size

        if not device:
            device = self.device

        experiences = random.sample(self.memory_buffer, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        is_state_terminals = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, is_state_terminals)

