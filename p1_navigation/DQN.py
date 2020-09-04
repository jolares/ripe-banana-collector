import numpy as np
import random

import torch
import torch.nn.functional as torch_func
import torch.optim as torch_optim

from p1_navigation import QNetwork
from p1_navigation.QNetwork import QNetwork
from p1_navigation.ReplayMemoryBuffer import ReplayMemoryBuffer
from p1_navigation.EpsilonGreedyExploration import EpsilonGreedyExplorationStrategy


### Default DQN Param Values

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Learning Hyper-params
ALPHA = 0.0005          # learning rate
GAMMA = 0.9             # discount factor

## Exploration Strategy
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99
EXPLORATION_STRATEGY_FN = EpsilonGreedyExplorationStrategy

## Deep Q-Network Config
UPDATE_INTERVAL = 5     # how often to update the network
TAU = 1e-3              # for soft update of target parameters
ADAM_OPTIMIZER_FN = torch_optim.Adam
MSE_LOSS_FN = torch_func.mse_loss

## Experience Replay Config
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
MIN_NUM_BATCHES = 5


def soft_update(online_q_network, target_q_network, tau):
    for target_param, local_param in zip(target_q_network.parameters(), online_q_network.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class DQN:
    """A Deep Q-Network agent that learns from interacting with an environment"""

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 q_network_hidden_layers_dims: tuple,
                 q_network_activation_fns: tuple,
                 alpha: float = ALPHA,
                 gamma: float = GAMMA,
                 epsilon_start: float = EPSILON_START,
                 epsilon_min: float = EPSILON_MIN,
                 epsilon_decay: float = EPSILON_DECAY,
                 tau: float = TAU,
                 buffer_size: int = BUFFER_SIZE,
                 batch_size: int = BATCH_SIZE,
                 min_num_batches: int = MIN_NUM_BATCHES,
                 optimizer_fn = ADAM_OPTIMIZER_FN,
                 loss_fn = MSE_LOSS_FN,
                 q_model_update_step_interval: int = 5,
                 q_network_update_fn = soft_update,
                 exploration_strategy_fn = EXPLORATION_STRATEGY_FN,
                 device: str = DEVICE,
                 seed = 0):
        """

        Args:
            state_size:
            action_size:
            q_network_hidden_layers_dims:
            q_network_activation_fns:
            alpha: the learning rate to be used by the optimizer.
            gamma: the discount rate.
            epsilon_start: the starting random action probability.
            epsilon_min: the minimum random action probability.
            epsilon_decay: the random action decay rate.
            tau:
            buffer_size:
            batch_size:
            min_num_batches:
            optimizer_fn:
            loss_fn:
            q_model_update_step_interval:
            q_network_update_fn:
            exploration_strategy_fn:
            device:
            seed:
        """

        self.seed = seed
        self.device = device
        self.seed = random.seed(seed)

        ## MDP & Interaction Tracking

        self.state_size = state_size
        self.action_size = action_size

        self.episode_idx = 0
        self.step_idx = 0

        ## Q-learning Hyper-Params

        self.alpha = alpha
        self.gamma = gamma
        self.exploration_strategy = exploration_strategy_fn(epsilon_start=epsilon_start,
                                                            epsilon_min=epsilon_min,
                                                            epsilon_decay=epsilon_decay)

        ## Q-Network Params

        self.online_q_network = QNetwork(input_dim=state_size,
                                         output_dim=action_size,
                                         hidden_layers_dims=q_network_hidden_layers_dims,
                                         activation_fns=q_network_activation_fns,
                                         device=device,
                                         seed=seed).to(device)

        self.target_q_network = QNetwork(input_dim=state_size,
                                         output_dim=action_size,
                                         hidden_layers_dims=q_network_hidden_layers_dims,
                                         activation_fns=q_network_activation_fns,
                                         device=device,
                                         seed=seed).to(device)

        self.tau = tau
        self.q_model_update_step_interval = q_model_update_step_interval
        self.q_network_update_fn = q_network_update_fn
        self.optimizer = optimizer_fn(self.online_q_network.parameters(), lr=alpha)
        self.loss = loss_fn

        ## Experience Replay Config

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.min_num_batches = min_num_batches
        self.replay_buffer = ReplayMemoryBuffer(buffer_size=buffer_size, batch_size=batch_size, seed=seed)

        self.min_memory_size_for_sampling = self.replay_buffer.batch_size * self.min_num_batches


    @property
    def epsilon(self):
        """The agent's current random action probability."""
        return self.exploration_strategy.epsilon


    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_state_terminals = experiences

        # Get target Q-Network's predicted max_action values for next_states
        q_targets_next = self.target_q_network(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute target Q-values for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - is_state_terminals))

        # Get the expected Q values from online/local model
        q_expected = self.online_q_network(states).gather(1, actions)

        # Compute loss
        loss = self.loss(q_expected, q_targets)

        ## Minimize loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def choose_action(self, state, device = DEVICE, epsilon_override: float = None):
        """Returns an action to be taken at a given state based on the current policy and exploration strategy."""
        return self.exploration_strategy.choose_action(model=self.online_q_network,
                                                       state=state, device=device,
                                                       epsilon_override=None)


    def step(self, state, action, reward, next_state, done):
        """
        Runs the learning procedure for one iteration of agent-env interaction.

        Args:
            state:
            action:
            reward:
            next_state:
            done:

        Returns:

        """
        # Store experience in Replay Memory
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) > self.min_memory_size_for_sampling:
            experiences = self.replay_buffer.sample()
            self.optimize_model(experiences)

        if np.sum(self.episode_idx) % self.q_model_update_step_interval == 0:
            self.update_target_q_network(online_q_network=self.online_q_network,
                                         target_q_network=self.target_q_network,
                                         tau=self.tau)

        # Decay Exploration Rate
        self.exploration_strategy.step()

        if done:
            self.episode_idx += 1
            return


    def update_target_q_network(self, online_q_network, target_q_network, tau = TAU):
        ## Update target Q-Network
        self.q_network_update_fn(online_q_network, target_q_network, tau)
