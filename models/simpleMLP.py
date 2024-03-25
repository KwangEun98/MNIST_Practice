import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, hidden_dims:list, input_dim:int = 28*28, output_dim:int = 10):
        """
        Simple MLP in MNIST

        Args:
            input_dim (int): input dimension (MNIST에서는 28*28)
            hidden_dims (list): the list of hidden dimension
            output_dim (int): output dimension (MNIST에서는 10)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.layer = nn.Sequential()
        init_dim = self.input_dim
        for idx, hidden_dimension in enumerate(self.hidden_dims):
            self.layer.add_module('layer_' + str(idx), 
                                  nn.Linear(init_dim, hidden_dimension))
            self.layer.add_module('activation layer',
                                  nn.ReLU())
            init_dim = hidden_dimension
        self.classifer_layer = nn.Linear(init_dim, output_dim)
        
    def forward(self, x):
        x = self.layer(x.view(-1, self.input_dim))
        x = self.classifer_layer(x)
        return x
