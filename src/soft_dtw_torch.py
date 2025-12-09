import torch

class SoftDTWTorch(torch.nn.Module):
    """
    PyTorch implementation of Soft-DTW 

    Computes a differentiable approximation of DTW using a soft-min
    operator controlled by the smoothing parameter gamma.
    """

    def __init__(self, gamma=1.0):
        """
        Parameters

        gamma : float
            Smoothing parameter. Lower values → closer to DTW,
            higher values → smoother, more differentiable alignment.
        """
        super().__init__()
        self.gamma = gamma

    def _softmin(self, a, b, c):
        """
        Smooth min operator over three values.

        Parameters

        a, b, c : torch.Tensor
            Neighboring values in the DP table.

        Returns

        torch.Tensor
            Soft-min computed via log-sum-exp.
        """
        stacked = torch.stack([a, b, c], dim=-1)
        return -self.gamma * torch.logsumexp(-stacked / self.gamma, dim=-1)

    def forward(self, A, B):
        """
        Compute the soft-DTW cost between sequences A and B.

        Parameters

        A, B : torch.Tensor, shape (T, d)
            Input time series.

        Returns

        torch.Tensor
            Soft-DTW alignment cost.
        """
        D = squared_euclidean_distances(A, B)  
        self.D = D

        T = D.size(0)
        R = torch.full((T+2, T+2), float('inf'), device=D.device, dtype=D.dtype)
        R[0,0] = 0.

        for i in range(1, T+1):
            Di = D[i-1]
            Ri_1 = R[i-1]
            Ri = R[i]
            for j in range(1, T+1):
                Ri[j] = Di[j-1] + self._softmin(
                    Ri_1[j], Ri_1[j-1], Ri[j-1]
                )

        self.R = R
        return R[T, T]

    def backward_pass(self):
        """
        Compute the backward dynamic programming pass to obtain
        the expected alignment matrix (E-matrix), which is used
        to derive gradients w.r.t. the input sequences.

        Returns

        torch.Tensor, shape (T, T)
            Expected alignment matrix.
        """
        D = torch.nn.functional.pad(self.D, (0,1,0,1), value=0)  
        R = self.R

        T = self.D.size(0)
        gamma = self.gamma

        E = torch.zeros((T+2, T+2), device=D.device, dtype=D.dtype)

        R[-1, :] = -float('inf')
        R[:, -1] = -float('inf')
        R[T+1, T+1] = R[T, T]
        E[T+1, T+1] = 1.

        for j in range(T, 0, -1):
            for i in range(T, 0, -1):
                a = torch.exp((R[i+1,j]   - R[i,j] - D[i,j-1]) / gamma)
                b = torch.exp((R[i,j+1]   - R[i,j] - D[i-1,j]) / gamma)
                c = torch.exp((R[i+1,j+1] - R[i,j] - D[i,j])   / gamma)

                E[i,j] = (
                    E[i+1,j] * a +
                    E[i,j+1] * b +
                    E[i+1,j+1] * c
                )

        E = E[1:T+1, 1:T+1]  

        self.E = E
        return E 

    def gradient(self, A, B):
        """
        Compute the gradient of soft-DTW with respect to A.

        Parameters

        A, B : torch.Tensor
            Input sequences.

        Returns

        torch.Tensor, shape (T, d)
            Gradient of soft-DTW w.r.t A.
        """
        E = self.E
        return jacobian_sq_euc(A, B, E)


def squared_euclidean_distances(A, B):
    """
    Compute the pairwise squared Euclidean distance matrix between
    sequences A and B.

    Parameters

    A, B : torch.Tensor, shape (T, d)

    Returns

    torch.Tensor, shape (T, T)
        Matrix of ||A_i - B_j||^2.
    """
    diff = A[:,None,:] - B[None,:,:]
    return torch.sum(diff * diff, dim=2)


def jacobian_sq_euc(A, B, E):
    """
    Compute the gradient of the squared Euclidean distance
    contribution weighted by the alignment matrix E.

    Parameters

    A, B : torch.Tensor
        Input sequences.
    E : torch.Tensor, shape (T, T)
        Expected alignment matrix.

    Returns

    torch.Tensor, shape (T, d)
        Gradient of soft-DTW w.r.t A.
    """
    T, d = A.size()
    diff = A[:,None,:] - B[None,:,:]        
    weighted = E[:,:,None] * diff * 2       
    G = weighted.sum(dim=1)                 
    return G
