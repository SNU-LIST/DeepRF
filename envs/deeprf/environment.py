import gym
import torch
from envs.deeprf.core import SLRExcitation


class DeepRFSLREXC20(gym.Env):
    def __init__(self, **kwargs):
        self.core = SLRExcitation(**kwargs)
        self.input_shape = self.core.input_shape
        self.action_space = self.core.action_space
        self.du = self.core.du
        self.max_amp = self.core.max_amp
        self.df = self.core.df
        self.sar_coef = kwargs.get('sar_coef', 0.0)
        self.ripple_coef = kwargs.get('ripple_coef', 1.0)
        self.max_mag = kwargs.get('max_mag', 0.9833)
        self.max_ripple = kwargs.get('max_ripple', 0.01)

    def __len__(self):
        return self.core.sampling_rate

    def reset(self):
        return self.core.reset()

    def step(self, actions):
        m = actions[..., 0]
        phi = actions[..., 1]
        Mt, done = self.core.step(m, phi)
        amp = ((torch.clamp(m, -1.0, 1.0) + 1.0) * self.core.max_amp * 1e+4 / 2).pow(2) * \
              self.du / self.core.sampling_rate * 1e+6  # (mG)^2sec
        rewards = -self.sar_coef * amp
        if done:
            Mt1 = Mt[:, 0, :1000, :]  # passband
            Mt2 = Mt[:, 0, 1000:, :]  # stopband
            mxy1 = torch.sqrt(torch.sum(Mt1[..., 0], dim=1) ** 2 + torch.sum(Mt1[..., 1], dim=1) ** 2) / 1000
            mxy2 = torch.max(torch.sqrt(Mt2[..., 0] ** 2 + Mt2[..., 1] ** 2), dim=1)[0]
            rewards += torch.clamp(mxy1, max=self.max_mag) - self.ripple_coef * torch.clamp(mxy2 - self.max_ripple, min=0.0)
        return Mt.permute(0, 2, 1, 3), rewards, done

