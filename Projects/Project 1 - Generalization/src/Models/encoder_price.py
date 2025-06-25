import torch
import torch.nn as nn
import torch.nn.functional as F


class PriceEncoder(nn.Module):
    """
    A flexible encoder for engineered price features with pluggable dropout strategies.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=[64, 32],
        output_dim=16,
        dropout_type="standard",  # options: "standard", "dropconnect", "none"
        dropout_p=0.3,
    ):
        """
        Arguments:
            input_dim (int): Number of input features
            hidden_dims (list): List of hidden layer sizes
            output_dim (int): Output latent dimension
            dropout_type (str): Type of dropout ("standard", "dropconnect", "none")
            dropout_p (float): Dropout probability
        """
        super().__init__()
        self.dropout_type = dropout_type
        self.dropout_p = dropout_p

        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Register DropConnect mask (if needed)
        if self.dropout_type == "dropconnect":
            self.register_buffer("dropconnect_mask", torch.ones_like(self.output_layer.weight))

    def apply_dropout(self, x):
        if self.dropout_type == "standard":
            return F.dropout(x, p=self.dropout_p, training=self.training)
        return x  # No-op for other types

    def dropconnect(self, weights):
        if not self.training:
            return weights
        mask = torch.bernoulli((1 - self.dropout_p) * torch.ones_like(weights))
        return weights * mask / (1 - self.dropout_p)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            x = self.apply_dropout(x)

        # Output projection — apply DropConnect if selected
        if self.dropout_type == "dropconnect":
            w = self.dropconnect(self.output_layer.weight)
            x = F.linear(x, w, self.output_layer.bias)
        else:
            x = self.output_layer(x)

        return x
