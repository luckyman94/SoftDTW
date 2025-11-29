import sys, os
sys.path.append(os.path.abspath(".."))

import numpy as np
from src.soft_dtw import SoftDTW
from scipy.optimize import minimize
from src.soft_dtw import SoftDTW
from src.soft_dtw import jacobian_sq_euc, squared_euclidean_distances

def softdtw_barycenter(X_list, Z_init, gamma=1.0, max_iter=50, tol=1e-3, weights=None):
    if X_list is None or len(X_list) == 0:
        raise ValueError("X_list must contain at least one time series.")
    if weights is None:
        weights = np.ones(len(X_list)) # uniform weights
    else:
        weights = np.array(weights)

    T, d = Z_init.shape

    def objective(Z_flat):
        Z = Z_flat.reshape(T, d)
        total_cost = 0.0
        total_grad = np.zeros_like(Z)

        for w, X in zip(weights, X_list):
            D = squared_euclidean_distances(Z, X)
            sdtw = SoftDTW(D, gamma=gamma)

            cost = sdtw.forward()
            E = sdtw.backward()

            #Calcul du gradient 
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


