import numpy as np

class SoftDTW:
    def __init__(self, D, gamma=1.0):
        """
        D : distance matrix (m x n), usually squared Euclidean distances
        """
        self.D = D
        self.gamma = gamma
        self.R_ = None   # forward matrix
        self.E_ = None   # backward matrix (gradient)
    

    # ----------------------------------------------------------------------
    # Soft-min over 3 values
    # ----------------------------------------------------------------------
    def _softmin3(self, a, b, c, gamma):
        a = -a / gamma
        b = -b / gamma
        c = -c / gamma
        max_val = max(a, b, c)
        tmp = np.exp(a - max_val) + np.exp(b - max_val) + np.exp(c - max_val)
        return -gamma * (np.log(tmp) + max_val)


    # ----------------------------------------------------------------------
    # FORWARD : compute soft-DTW value
    # ----------------------------------------------------------------------
    def compute(self):
        """
        Compute soft-DTW by dynamic programming (Python version).

        Returns
        -------
        float : Soft-DTW discrepancy
        """
        D = self.D
        m, n = D.shape

        # Allocate (m+2 x n+2) as in original Cython code
        R = np.full((m+2, n+2), np.inf)
        R[0, 0] = 0.0

        gamma = self.gamma

        # Forward DP
        for i in range(1, m+1):
            for j in range(1, n+1):
                R[i, j] = D[i-1, j-1] + self._softmin3(
                    R[i-1, j],      # insertion
                    R[i-1, j-1],    # match
                    R[i, j-1],      # deletion
                    gamma
                )

        self.R_ = R  # save for backward
        return R[m, n]


    # ----------------------------------------------------------------------
    # BACKWARD : gradient wrt distance matrix D (Python version)
    # ----------------------------------------------------------------------
    def _soft_dtw_grad(self):
        """
        Compute gradient wrt the distance matrix D.
        Equivalent to Cython version in Blondel et al.
        """
        D = np.pad(self.D, ((0,1),(0,1)), mode='constant')  # to make (m+1)x(n+1)
        R = self.R_
        
        m = D.shape[0] - 1
        n = D.shape[1] - 1
        gamma = self.gamma

        # Allocate E
        E = np.zeros((m+2, n+2), dtype=np.float64)

        # Initialization as in original Soft-DTW
        for i in range(1, m+1):
            D[i-1, n] = 0
            R[i, n+1] = -np.inf

        for j in range(1, n+1):
            D[m, j-1] = 0
            R[m+1, j] = -np.inf

        E[m+1, n+1] = 1.0
        R[m+1, n+1] = R[m, n]
        D[m, n] = 0

        # Backward DP
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

        self.E_ = E
        return E[1:m+1, 1:n+1] 
    


    def squared_euclidean_distances(self,A, B):
        diff = A[:, None, :] - B[None, :, :]
        return np.sum(diff * diff, axis=2)
    

def jacobian_sq_euc(self,A, B, E):
    T, d = A.shape
    G = np.zeros_like(A)

    # formula: G[i] = Σ_j E[i,j] * 2*(A[i]-B[j])
    for i in range(T):
        for j in range(T):
            G[i] += E[i, j] * 2 * (A[i] - B[j])

    return G