
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----------------------
# Utility Functions
# ----------------------
def train_val_split(X, y, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - val_ratio))
    return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]

def standardize(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / (std + 1e-8), mean, std

def compute_sharpe(preds, targets):
    excess = preds - targets
    return np.mean(excess) / (np.std(excess) + 1e-8)

def plot_losses(train_losses, val_losses, config_name="Model"):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(f"Loss - {config_name}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------
# PriceEncoder Model
# ----------------------
class PriceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_type="standard", dropout_p=0.3):
        super().__init__()
        self.dropout_type = dropout_type
        self.dropout_p = dropout_p
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h))
            prev_dim = h
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def apply_dropout(self, x):
        if self.dropout_type == "standard":
            return F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def forward(self, x):
        for layer in self.layers:
            x = self.apply_dropout(F.relu(layer(x)))
        return self.output_layer(x)

# ----------------------
# Training Loop
# ----------------------
def train_model(X, y, config, num_epochs=50, lr=1e-3, val_ratio=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_std, _, _ = standardize(X)
    X_train, X_val, y_train, y_val = train_val_split(X_std, y, val_ratio)

    model = PriceEncoder(
        input_dim=X.shape[1],
        hidden_dims=config["hidden_dims"],
        output_dim=config["output_dim"],
        dropout_type=config["dropout_type"],
        dropout_p=config["dropout_p"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        inputs = torch.tensor(X_train).float().to(device)
        targets = torch.tensor(y_train).float().to(device)

        preds = model(inputs)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_inputs = torch.tensor(X_val).float().to(device)
            val_targets = torch.tensor(y_val).float().to(device)
            val_preds = model(val_inputs)
            val_loss = criterion(val_preds, val_targets)
            val_losses.append(val_loss.item())

    final_preds = model(torch.tensor(X_val).float().to(device)).detach().cpu().numpy()
    sharpe = compute_sharpe(final_preds, y_val)

    plot_losses(train_losses, val_losses, config_name=str(config["hidden_dims"]))
    print(f"Final Sharpe Ratio: {sharpe:.4f}")
    return model

# ----------------------
# Demo: Run It
# ----------------------
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randn(500, 16)
    y = np.random.randn(500, 1)

    config = {
        "hidden_dims": [64, 32],
        "output_dim": 1,
        "dropout_type": "standard",
        "dropout_p": 0.2
    }

    train_model(X, y, config)
