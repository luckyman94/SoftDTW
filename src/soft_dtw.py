import numpy as np

class SoftDTW:
    """
    NumPy implementation of Soft-DTW.

    Computes a smoothed version of Dynamic Time Warping using a soft-min
    operator controlled by the smoothing parameter gamma.
    """

    def __init__(self, D, gamma=1.0):
        """
        Initialize the Soft-DTW solver.

        Parameters
        D : np.ndarray
            Pairwise distance matrix between the two sequences.
        gamma : float
            Smoothing parameter (lower = closer to classical DTW, higher = smoother).
        """
        self.D = D
        self.gamma = gamma
        self._R = None   
        self._E = None   
    
    def _softmin(self, a, b, c, gamma):
        """
        Smooth minimum over three values using a log-sum-exp formulation.

        Parameters
        a, b, c : float
            Neighboring DP values.
        gamma : float
            Smoothing parameter.

        Returns
        float
            Soft-min approximation of min(a, b, c).
        """
        a = -a / gamma
        b = -b / gamma
        c = -c / gamma
        max_val = max(a, b, c)
        tmp = np.exp(a - max_val) + np.exp(b - max_val) + np.exp(c - max_val)
        return -gamma * (np.log(tmp) + max_val)
    
    def forward(self):
        """
        Compute the forward dynamic programming pass of Soft-DTW.

        Parameters
        None

        Returns
        float
            Soft-DTW alignment cost.
        """
        D = self.D
        m, n = D.shape

        R = np.full((m+2, n+2), np.inf)
        R[0, 0] = 0.0

        for i in range(1, m+1):
            for j in range(1, n+1):
                R[i, j] = D[i-1, j-1] + self._softmin(
                    R[i-1, j],      
                    R[i-1, j-1],    
                    R[i, j-1],      
                    self.gamma
                )
        self._R = R 
        return R[m, n]

    def backward(self):
        """
        Compute the backward pass of Soft-DTW to obtain the expected 
        alignment matrix (E-matrix), used to compute gradients.

        Parameters
        None

        Returns

        np.ndarray
            Expected alignment matrix of shape (m, n).
        """
        D = np.pad(self.D, ((0,1),(0,1)), mode='constant')  
        R = self._R
        
        m = D.shape[0] - 1
        n = D.shape[1] - 1
        gamma = self.gamma

        E = np.zeros((m+2, n+2), dtype=np.float64)

        for i in range(1, m+1):
            D[i-1, n] = 0
            R[i, n+1] = -np.inf

        for j in range(1, n+1):
            D[m, j-1] = 0
            R[m+1, j] = -np.inf

        E[m+1, n+1] = 1.0
        R[m+1, n+1] = R[m, n]
        D[m, n] = 0

        for j in range(n, 0, -1):
            for i in range(m, 0, -1):
                a = np.exp((R[i+1, j]   - R[i, j] - D[i,   j-1]) / gamma)
                b = np.exp((R[i,   j+1] - R[i, j] - D[i-1, j  ]) / gamma)
                c = np.exp((R[i+1, j+1] - R[i, j] - D[i,   j  ]) / gamma)

                E[i, j] = (
                    E[i+1, j] * a +
                    E[i, j+1] * b +
                    E[i+1, j+1] * c
                )

        self._E = E
        return E[1:m+1, 1:n+1] 
    


def squared_euclidean_distances(A, B):
    """
    Compute pairwise squared Euclidean distances between sequences A and B.

    Parameters
    A : np.ndarray
        First sequence, shape (T, d).
    B : np.ndarray
        Second sequence, shape (T, d).

    Returns
    np.ndarray
        Distance matrix of shape (T, T).
    """
    diff = A[:, None, :] - B[None, :, :]
    return np.sum(diff * diff, axis=2)
    

def jacobian_sq_euc(A, B, E):
    """
    Compute the gradient of Soft-DTW with respect to A using the E-matrix.

    Parameters
    A : np.ndarray
        Input sequence A, shape (T, d).
    B : np.ndarray
        Input sequence B, shape (T, d).
    E : np.ndarray
        Expected alignment matrix, shape (T, T).

    Returns
    np.ndarray
        Gradient of the Soft-DTW loss with respect to A.
    """
    T, d = A.shape
    G = np.zeros_like(A)

    for i in range(T):
        for j in range(T):
            G[i] += E[i, j] * 2 * (A[i] - B[j])

    return G
