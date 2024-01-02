import torch.nn as nn


class Model(nn.Module):
    def __init__ (self, n_input, n_output, n_hidden, n_layers):
        super(Model, self).__init__()

        activation = nn.Tanh # twice differentiable activation function

        self.pinn_start = nn.Sequential(*[
            nn.Linear(n_input, n_hidden),
            activation()]
        )

        self.pinn_hidden = nn.Sequential(*[
            nn.Sequential(*[nn.Linear(n_hidden, n_hidden), activation()]) for _ in range(n_layers - 1)])
        
        self.pinn_end = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.pinn_start(x)
        x = self.pinn_hidden(x)
        x = self.pinn_end(x)
        return x
