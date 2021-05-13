"""code modified from https://github.com/namalkanti/bloch-simulator-python"""
import math
from collections import Iterable
import torch


class BlochSimulator:
    def __init__(self, **kwargs):
        """Bloch simulation for B1, gradients and off-resonance.

            Args::
                gamma: gyromagnetic ratio (Hz/T)
                M0: array of initial magnetization [N, P, 3]
                ts: time duration of each B1 and gradient point (sec) [T]
                T1: T1 relaxation time (sec) [P]
                T2: T2 relaxation time (sec) [P]
                df: array of off-resonance frequencies (Hz) [N]
                dp: array of spatial positions (m) [P, 3]

        """
        # Define constants
        keys = ['gamma', 'M0', 'ts', 'T1', 'T2', 'df', 'dp']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for k in keys:
            msg = "BlochSimulator must have '{}' as its argument".format(k)
            v = kwargs.get(k, None)
            assert v is not None, msg

            if isinstance(v, Iterable):
                v = torch.FloatTensor(v).to(self.device)
            self.__dict__.update({k: v})

        # Initialize simulator
        _ = self.reset()

    def reset(self):
        """Initialize magnetization state vector
        """
        self.time = 0.0
        self.idx = 0
        self.M = self.M0.clone().unsqueeze(0)  # [1, N, P, 3]
        return self.M

    def step(self, B1, G):
        """Simulate single time step of the Bloch equation

            Args::
                B1: complex RF pulse for given time t (T) [B, 2]
                G: 3-dimensional gradient (T/m) [B, 3]

            Returns::
                Mt: batch of magnetization at current time step [B, N, P, 3]
                done: boolean flag for denoting the end of simulation

        """
        msg = "arguments must be provided by batch"
        assert B1.ndim == 2 and G.ndim == 2, msg
        assert B1.shape[-1] == 2, "B1 field must be complex"
        assert G.shape[-1] == 3, "spatial gradient must be 3-dimensional"

        # Convert SI units
        ts = self.ts[self.idx]  # (sec)
        gamma = 2 * math.pi * self.gamma  # (rad/sec·T)
        df = 2 * math.pi * self.df  # (rad/sec)
        B1_real = B1[..., 0]  # (T)
        B1_imag = B1[..., 1]  # (T)

        # Rotation angles
        nx = -B1_real[..., None, None] * gamma * ts  # (rad)
        ny = B1_imag[..., None, None] * gamma * ts  # (rad)
        nz = (G.unsqueeze(1) * self.dp.unsqueeze(0) * gamma).sum(-1)  # (rad/sec)
        nz = nz.unsqueeze(1) + df.unsqueeze(0).unsqueeze(-1)  # (rad/sec)
        nz = -nz * ts  # [B, N, P], (rad)

        # Cayley-Klein parameters
        phi = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2)  # [B, N, P]
        sin_phi = torch.sin(phi / 2) / phi
        # sin_phi[sin_phi != sin_phi] = 0.0  # Convert nans to zero
        cos_phi = torch.cos(phi / 2)
        ar = cos_phi
        ai = -nz * sin_phi
        br = ny * sin_phi
        bi = -nx * sin_phi

        # Rotation matrix
        rot_mat = torch.zeros((*phi.shape, 3, 3), device=self.device)  # [B, N, P, 3, 3]
        rot_mat[..., 0, 0] = ar ** 2 - ai ** 2 - br ** 2 + bi ** 2
        rot_mat[..., 1, 0] = -2 * ar * ai - 2 * br * bi
        rot_mat[..., 2, 0] = -2 * ar * br + 2 * ai * bi
        rot_mat[..., 0, 1] = 2 * ar * ai - 2 * br * bi
        rot_mat[..., 1, 1] = ar ** 2 - ai ** 2 + br ** 2 - bi ** 2
        rot_mat[..., 2, 1] = -2 * ai * br - 2 * ar * bi
        rot_mat[..., 0, 2] = 2 * ar * br + 2 * ai * bi
        rot_mat[..., 1, 2] = 2 * ar * bi - 2 * ai * br
        rot_mat[..., 2, 2] = ar ** 2 + ai ** 2 - br ** 2 - bi ** 2

        Mt = self.M.view(self.M.size(0), -1, 3).unsqueeze(-1)
        Mt = Mt.expand(rot_mat.size(0), -1, -1, -1)
        Mt = Mt.reshape(-1, 3, 1)  # [B x N x P, 3, 1]
        rot_mat = rot_mat.view(-1, 3, 3)  # [B x N x P, 3, 3]
        Mt = torch.bmm(rot_mat, Mt)  # [B x N x P, 3, 1]

        # Time decay matrix
        e1 = torch.exp(-ts / self.T1)
        e2 = torch.exp(-ts / self.T2)
        dec_mat = torch.zeros((len(e1), 3, 3), device=self.device)  # [P, 3, 3]
        dec_mat[:, 0, 0] = e2
        dec_mat[:, 1, 1] = e2
        dec_mat[:, 2, 2] = e1
        dec_vec = torch.zeros((len(e1), 3), device=self.device)  # [P, 3]
        dec_vec[:, 2] = 1 - e1

        size = (phi.size(0) * phi.size(1) * phi.size(2), -1, -1, -1)
        dec_mat = dec_mat.unsqueeze(0).expand(size)
        dec_mat = dec_mat.view(-1, 3, 3)  # [B x N x P, 3, 3]
        Mt = torch.bmm(dec_mat, Mt)  # [B x N x P, 3, 1]
        Mt = Mt.view(*phi.shape, 3)  # [B, N, P, 3]
        dec_vec = dec_vec[None, None, ...]  # [1, 1, P, 3]
        Mt = Mt + dec_vec  # [B, N, P, 3]

        # Step forward and check if done
        self.M = Mt
        self.time += self.ts[self.idx]
        self.idx += 1
        done = (self.idx == len(self.ts))

        return Mt, done
