import sys, os
sys.path.append(os.path.abspath(".."))

import numpy as np
from src.soft_dtw import SoftDTW
from scipy.optimize import minimize
from src.soft_dtw import SoftDTW
from src.soft_dtw import jacobian_sq_euc, squared_euclidean_distances

def softdtw_barycenter(X_list, Z_init, gamma=1.0, max_iter=50, tol=1e-3, weights=None):
    """
    Compute the Soft-DTW barycenter of a set of time series.

    The barycenter is defined as the sequence Z that minimizes the weighted
    sum of Soft-DTW distances to all sequences in X_list.

    Parameters
    
    X_list : list of np.ndarray
        List of time series to average. Each array has shape (T, d).
    Z_init : np.ndarray
        Initial guess for the barycenter, shape (T, d).
    gamma : float
        Soft-DTW smoothing parameter.
    max_iter : int
        Maximum number of L-BFGS-B iterations.
    tol : float
        Convergence tolerance for the optimizer.
    weights : list or np.ndarray, optional
        Weights for each time series (default: uniform).

    Returns
    np.ndarray
        Estimated barycenter Z, shape (T, d).
    """
    if X_list is None or len(X_list) == 0:
        raise ValueError("X_list must contain at least one time series.")
    if weights is None:
        weights = np.ones(len(X_list)) # uniform weights
    else:
        weights = np.array(weights)

    T, d = Z_init.shape

    def objective(Z_flat):
        """
        Objective function for L-BFGS-B.

        Computes:
        - the weighted sum of Soft-DTW costs,
        - the corresponding gradient with respect to Z.

        Returns a tuple (cost, gradient_flattened).
        """
        Z = Z_flat.reshape(T, d)
        total_cost = 0.0
        total_grad = np.zeros_like(Z)

        for w, X in zip(weights, X_list):
            D = squared_euclidean_distances(Z, X)
            sdtw = SoftDTW(D, gamma=gamma)

            cost = sdtw.forward()
            E = sdtw.backward()

            # Calcul du gradient 
            G = jacobian_sq_euc(Z, X, E)

            total_cost += w * cost
            total_grad += w * G
        
        return total_cost, total_grad.ravel()
    
    res = minimize(
        fun=objective,
        x0=Z_init.ravel(),
        method="L-BFGS-B",
        jac=True,
        tol=tol,
        options={"maxiter": max_iter, "disp": False},
    )

    return res.x.reshape(T, d)
