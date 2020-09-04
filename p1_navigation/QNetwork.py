import torch
from torch import nn as nn
from torch.nn import functional as torch_func


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HIDDEN_LAYERS_DIMS = (64, 64)
ACTIVATION_FNS = (torch_func.relu, torch_func.relu)


class QNetwork(nn.Module):
    """Actor (Policy) Q-Network Model."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layers_dims: tuple = HIDDEN_LAYERS_DIMS,
                 activation_fns = ACTIVATION_FNS,
                 device: str = DEVICE,
                 seed: int = 0):
        """Initializes Q-Network parameters & Builds Model.
        
        Args:
            input_dim: dimension of the model's input (i.e. dimension of state-observations)
            output_dim: dimension of the model's output (i.e. dimension of action-space)
            hidden_layers_dims: dimensions of hidden layers, ordered from first to last.
            seed: seed for random number generation.
        """
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.device = torch.device(device)
        self.to(self.device)

        self.activation_fns = activation_fns

        ## Initialize Neural-Network Layers

        self.input_layer = nn.Linear(input_dim, hidden_layers_dims[0])

        # Initialize hidden layers
        self.hidden_layers = nn.ModuleList()
        for layer_dim in range(len(hidden_layers_dims) - 1):
            hidden_layer = nn.Linear(hidden_layers_dims[layer_dim], hidden_layers_dims[layer_dim + 1])
            self.hidden_layers.append(hidden_layer)

        self.output_layer = nn.Linear(hidden_layers_dims[-1], output_dim)


    def forward(self, state):
        """Builds a Forward-Pass Neural-Network which maps 'state' to 'actions' values."""

        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        x = self.activation_fns[0](self.input_layer(x))

        for layer_idx, layer_fn in enumerate(self.hidden_layers):
            x = self.activation_fns[layer_idx](layer_fn(x))

        return self.output_layer(x)
