import gym
import numpy as np
import torch
from envs.simulator import BlochSimulator
from settings import INF


class SLRExcitation:
    def __init__(
        self,
        b1_range=(1.0, 1.0, 1),
        off_resonance_range=(1e-8, 1e-8, 1),  # warning: only one zero will occurs NaN!
        du=2.56e-3,
        sampling_rate=256,
        pos_range1=(-1300, 1300, 201),
        pos_range2=(-32000, -1866, 800),
        pos_range3=(1866, 32000, 800),
        **kwargs
    ):
        msg = "start, end, and number of points must be specified"
        assert len(b1_range) == 3, msg
        assert len(off_resonance_range) == 3, msg

        pos1 = np.linspace(*pos_range1) / 42.5775 / 40 / 10
        pos2 = np.linspace(*pos_range2) / 42.5775 / 40 / 10
        pos3 = np.linspace(*pos_range3) / 42.5775 / 40 / 10

        # Create simulator
        gamma = 42.5775 * 1e6  # (Hz/T)
        dt = du / sampling_rate  # (sec)
        df = np.linspace(*off_resonance_range)
        dp = np.zeros((len(pos1) + len(pos2) + len(pos3), 3))
        dp[..., 2] = np.concatenate((pos1, pos2, pos3)) * 1e-2  # (m)
        M0 = np.zeros((len(df), dp.shape[0], 3))
        M0[..., 2] = 1.0

        self.simulator = BlochSimulator(
            gamma=gamma,
            ts=np.ones(sampling_rate) * dt,
            T1=[INF],
            T2=[INF],
            df=df,
            dp=dp,
            M0=M0
        )

        # Define constants
        self.b1_range = torch.linspace(*b1_range)
        self.du = du
        self.sampling_rate = sampling_rate
        self.max_amp = 0.2 * 1e-4  # (T)
        self.df = dp[..., 2] * gamma * 40 * 1e-3

        # Define action space
        self.input_shape = (M0.shape[1], M0.shape[0], M0.shape[2])  # (1801, 1, 3)
        self.action_space = gym.spaces.Box(
            # np.array([-1.0, -1.0]),
            # np.array([1.0, 1.0]),
            np.array([-INF, -INF]),
            np.array([INF, INF]),
            dtype=np.float64
        )

    def reset(self):
        return self.simulator.reset().permute(0, 2, 1, 3)  # (1, P, N, 3)

    def step(self, m, phi):
        # Clip and rescale input values
        m_clip = torch.clamp(m, -1.0, 1.0)
        m_scaled = (m_clip + 1.0) * self.max_amp / 2
        # p_clip = torch.clamp(phi, -1.0, 1.0)
        p_clip = phi
        p_scaled = p_clip * np.pi

        # Convert to complex value
        b1_real = m_scaled * torch.cos(p_scaled)
        b1_real = b1_real.unsqueeze(-1) * self.b1_range.unsqueeze(0).to(m)
        b1_imag = m_scaled * torch.sin(p_scaled)
        b1_imag = b1_imag.unsqueeze(-1) * self.b1_range.unsqueeze(0).to(m)
        B1 = torch.stack([b1_real.view(-1), b1_imag.reshape(-1)], dim=-1)

        # 40 mT/m == 4 G/cm z-gradient
        G = torch.zeros(B1.size(0), 3)
        G[..., 2] = 4 * 1e-2  # (T/m)
        Mt, done = self.simulator.step(B1, G.to(m))

        if done:  # rephasing gradient
            self.simulator.idx -= 1
            G[..., 2] = -4 * 1e-2 * self.sampling_rate / 2.0
            Mt, _ = self.simulator.step(torch.zeros_like(B1).to(m), G.to(m))

        return Mt, done

