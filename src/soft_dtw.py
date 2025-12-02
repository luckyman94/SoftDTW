import numpy as np

class SoftDTW:
    def __init__(self, D, gamma=1.0):
        self.D = D
        self.gamma = gamma
        self._R = None   
        self._E = None   
    
    def _softmin(self, a, b, c, gamma):
        a = -a / gamma
        b = -b / gamma
        c = -c / gamma
        max_val = max(a, b, c)
        tmp = np.exp(a - max_val) + np.exp(b - max_val) + np.exp(c - max_val)
        return -gamma * (np.log(tmp) + max_val)
    
    def forward(self):
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
        diff = A[:, None, :] - B[None, :, :]
        return np.sum(diff * diff, axis=2)
    

def jacobian_sq_euc(A, B, E):
    T, d = A.shape
    G = np.zeros_like(A)

    # formula: G[i] = Σ_j E[i,j] * 2*(A[i]-B[j])
    for i in range(T):
        for j in range(T):
            G[i] += E[i, j] * 2 * (A[i] - B[j])

    return G