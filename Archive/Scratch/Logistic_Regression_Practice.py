import numpy as np

# ------------------------------------------
# Locally Weighted Logistic Regression (Newton-Raphson)
# ------------------------------------------
# This script implements one iteration of the Newton-Raphson update
# for a locally-weighted logistic regression model.
# Each training example is assigned a weight based on its distance to a specific query point using a Gaussian kernel.
# The update computes the weighted log-likelihood gradient and Hessian, then performs a second-order Newton update.
# ------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_weights(X, query_point, tau):

#   Compute Gaussian weights for each training example based on distance to the query point.

    diff = X - query_point
    norms_squared = np.sum(diff ** 2, axis=1)
    weights = np.exp(-norms_squared / (2 * tau ** 2))
    return weights

def newton_step(X, y, theta, query_point, tau=1.0, lambda_reg=0.0001):
  
#   Perform one Newton-Raphson update for locally-weighted logistic regression.

#    Parameters:
#        X (m x n): Design matrix
#        y (m,): Target labels (0 or 1)
#        theta (n,): Current parameter vector
#        query_point (n,): The point around which local weighting is computed
#        tau (float): Bandwidth for Gaussian weights
#        lambda_reg (float): L2 regularization strength

#    Returns:
#        theta_new (n,): Updated parameter vector
 
    m, n = X.shape
    w = compute_weights(X, query_point, tau)

    # Compute predictions
    h = sigmoid(X @ theta)

    # Compute gradient
    z = w * (y - h)  # Element-wise multiplication
    gradient = X.T @ z - lambda_reg * theta

    # Compute diagonal matrix D
    D_diag = -w * h * (1 - h)
    D = np.diag(D_diag)

    # Compute Hessian
    H = X.T @ D @ X - lambda_reg * np.eye(n)

    # Newton-Raphson update
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        raise ValueError("Hessian is singular or ill-conditioned.")

    theta_new = theta - H_inv @ gradient
    return theta_new

# Example usage
if __name__ == "__main__":
    # Sample 2D dataset
    X = np.array([[1, 1],
                  [3, 2]])
    y = np.array([1, 0])
    theta = np.zeros(2)
    query_point = np.array([1, 2])
    tau = 1.0

    theta_updated = newton_step(X, y, theta, query_point, tau)

    print("Updated theta:", theta_updated)