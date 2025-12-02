import torch

class SoftDTWTorch(torch.nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def _softmin(self, a, b, c):
        stacked = torch.stack([a, b, c], dim=-1)
        return -self.gamma * torch.logsumexp(-stacked / self.gamma, dim=-1)

    def forward(self, A, B):
        D = squared_euclidean_distances(A, B)  # (T,T)
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
        D = torch.nn.functional.pad(self.D, (0,1,0,1), value=0)  # (T+1,T+1)
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
        E = self.E
        return jacobian_sq_euc(A, B, E)


def squared_euclidean_distances(A, B):
    diff = A[:,None,:] - B[None,:,:]
    return torch.sum(diff * diff, dim=2)


def jacobian_sq_euc(A, B, E):
    T, d = A.size()
    diff = A[:,None,:] - B[None,:,:]        
    weighted = E[:,:,None] * diff * 2       
    G = weighted.sum(dim=1)                 
    return G
