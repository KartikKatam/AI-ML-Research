import numpy as np
import matplotlib.pyplot as plt

# Custom Train/Validation Split
def train_val_split(X, y, val_ratio=0.2, seed=42):
    """
    Shuffles and splits input arrays into training and validation sets.

    Args:
        X (np.ndarray): Features
        y (np.ndarray): Targets
        val_ratio (float): Fraction of data used for validation
        seed (int): RNG seed for reproducibility

    Returns:
        Tuple: X_train, X_val, y_train, y_val
    """
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


# Standardize Features
def standardize(X):
    """
    Standardizes features to zero mean and unit variance. Helps with multimodal data analysis. 

    Args:
        X (np.ndarray): Original input data

    Returns:
        Tuple: standardized X, mean, std
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8), mean, std


# Compute Sharpe Ratio - a measure of risk-adjusted return
def compute_sharpe(preds, targets):
    """
    Computes the Sharpe ratio from predicted and true returns.

    Args:
        preds (np.ndarray): Predicted returns
        targets (np.ndarray): True returns

    Returns:
        float: Sharpe ratio
    """
    excess = preds - targets
    mean_excess = np.mean(excess)
    std_excess = np.std(excess)
    return mean_excess / (std_excess + 1e-8)


# Plot Loss Curves
def plot_losses(train_losses, val_losses, config_name="Model"):
    """
    Plots training and validation loss curves over epochs (one run through of the taining data).

    Args:
        train_losses (list): Training loss per epoch
        val_losses (list): Validation loss per epoch
        config_name (str): Model description (used in title)
    """
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title(f"Loss Curve - {config_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
