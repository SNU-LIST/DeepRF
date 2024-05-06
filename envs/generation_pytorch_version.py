"""
New version for the implementation of the Generation Module from the DeepRF Paper:

Shin, D., Kim, Y., Oh, C. et al. Deep reinforcement learning-designed radiofrequency waveform in MRI. 
Nat Mach Intell 3, 985–994 (2021). https://doi.org/10.1038/s42256-021-00411-1

The original code for this module was released by the authors and was written using both PyTorch and TensorFlow. 
Please refer to the 'generation.py' file in this repository.

This version provides a complete reimplementation of the Generation Module, based only PyTorch for improved code consistency and maintainability.
"""



from os import path
import sys
sys.path.append(path.join(path.dirname(__file__), '..'))
import os
import numpy as np
import time
import argparse
from ast import literal_eval as make_tuple
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import envs
import math
import matplotlib.pyplot as plt
from utils.logger import Logger
from utils.summary import EvaluationMetrics
from scipy.io import loadmat
from collections import OrderedDict
from network import SharedNetwork 

# %% arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='activated GPU number')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--el', type=int, default=32, help='episode length')
parser.add_argument('--amp', type=float, default=1e-3, help='amplitude scaling')  # 1e-3
parser.add_argument('--ph', type=float, default=1e+1, help='phase scaling')  # 1e+1
parser.add_argument('--hss', type=int, default=256, help='length of hidden state in GRU')
parser.add_argument('--batch', type=int, default=256, help='batch size (# episodes)')
parser.add_argument('--mb', type=int, default=2048, help='mini batch size (for 1 epoch)')
parser.add_argument('--v_hs', type=str, default='(256,128,64,32)', help='network structure')
parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
parser.add_argument('--lmbda', type=float, default=0.95, help='lambda for GAE')
parser.add_argument('--eps', type=float, default=0.1, help='args.eps for PPO')
parser.add_argument('--epochs', type=int, default=4, help='number of epochs for gradient-descent')
parser.add_argument('--max', type=int, default=300, help='maximum number of iterations')
parser.add_argument('--kl', type=float, default=0.01, help='target KL value for early stopping')
parser.add_argument('--du', type=float, default=2.56, help='duration of pulse in ms')
parser.add_argument('--w_v', type=float, default=1.0, help='value loss weight')
parser.add_argument('--amp_std', type=float, default=0.03, help='fixed amplitude standard deviation')
parser.add_argument('--ph_std', type=float, default=0.05, help='fixed phase standard deviation')
parser.add_argument('--seed', type=int, default=1003, help='random seed')
parser.add_argument('--grad', type=float, default=1000.0, help='l2-norm for gradient clipping')
parser.add_argument('--save', type=int, default=300, help='save period in iterations')
parser.add_argument("--tag", type=str, default='ppo_rnn_exc_21')
parser.add_argument("--log_level", type=int, default=10)
parser.add_argument("--debug", "-d", action="store_true")
parser.add_argument("--quiet", "-q", action="store_true")
parser.add_argument("--sampling_rate", type=int, default=256)
parser.add_argument("--log_step", type=int, default=10)
parser.add_argument("--env", type=str, default="Exc-v11")
args = parser.parse_args()

# %% preparation step
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)



# constants
output_size = 2  # (magnitude, phase)
EPS = 1e-8

# parsing arguments
ts = float(args.du) * 1e-3 / (float(args.sampling_rate))
max_rad = 2 * np.pi * 42.577 * 1e+6 * 0.2 * 1e-4 * ts

# fix random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


# load reference pulse
device = 'cuda' if torch.cuda.is_available() else 'cpu'
preset = loadmat('../data/conv_rf/SLR_exc.mat')
ref_pulse = torch.unsqueeze(torch.from_numpy(np.array(preset['result'], dtype=np.float32)), dim=0).to(device)


start_t = time.time()
best_SAR = 1e+10
best_RF = 0
best_ind = 0


logger = Logger('AINV', args)

# Create summary statistics
info = EvaluationMetrics([
    'Rew/Mean',
    'SAR/Mean',
    'Mz1/Mean',
    'Mz2/Mean',
    'Rew/Best',
    'SAR/Best',
    'Mz1/Best',
    'Mz2/Best',
    'Time',
])

class PPO:
    """
    Proximal Policy Optimization Agent Initialisation for Reinforcement Learning Training.

    This class implements the PPO algorithm for training the RL agent. 
    It initializes the agent with a shared neural network model and provides methods 
    for generating action distributions and performing policy updates based on obtained rollouts.

    Attributes:
        model (SharedNetwork): The shared neural network model used by the agent to model the policy and value function.
        optimizer (torch.optim.Adam): The optimizer used for updating the model parameters.

    """

    def __init__(self):
        """
        Initializes the PPO agent with a shared neural network model and optimizer.

        The neural network model is initialized using the SharedNetwork class, 
        and the Adam optimizer is used for updating the model parameters.
        """
        self.model = SharedNetwork(hidden_sizes=(256,128,64,32)).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def get_action(self, rnn_in, state_in):
        """
        Generates action distributions and value estimates based on the given input data and state.

        Args:
            rnn_in (torch.Tensor): Input data for the recurrent neural network.
            state_in (torch.Tensor): Initial state for the recurrent neural network.

        Returns:
            tuple: A tuple containing:
                - pi_clip (torch.Tensor): Clipped action distributions.
                - p_pi_a (torch.Tensor): Probability density of the selected actions.
                - value_out (torch.Tensor): Value estimates for the input data.
                - state_out (torch.Tensor): Updated state of the recurrent neural network.
        """
        output, state_out = self.model(rnn_in, state_in)
        value_out = output[:, 2]
        mean = torch.cat((torch.exp(output[:, 0]).unsqueeze(-1), output[:, 1].unsqueeze(-1)), dim=1)

        std_d = torch.mul(torch.tensor([max_rad * (1 / args.amp), 2 * np.pi * (1 / args.ph)], device=device),
                          torch.tensor([args.amp_std, args.ph_std], device=device))

        r = torch.empty_like(mean).normal_(mean=0.0, std=1.0).to(device)
        pi = mean + r * std_d                                               #Sample (A_t+1, P_t+1) from gaussian distributions

        pi_clip = torch.clamp(pi, min=torch.Tensor([EPS, -np.inf]).to(device),
                              max=torch.Tensor([max_rad * (1 / args.amp), np.inf]).to(device))

        middle1 = torch.exp(torch.multiply((pi_clip - mean).pow(2), (-1.0 / (2.0 * std_d.pow(2))))).prod(dim=-1,
                                                                                                           keepdim=True)
        p_pi_a = middle1 * (1 / (torch.sqrt(torch.tensor(np.pi * 2.0, device=device)) * std_d)).prod(dim=-1,
                                                                                                        keepdim=True) + EPS

        return pi_clip, p_pi_a.squeeze(dim=-1), value_out, state_out

    def train(self, rnn_in, state_in, adv, prob, ret):
        """
        Performs policy updates based on obtained rollouts and advantages.

        Args:
            rnn_in (torch.Tensor): Input data for the recurrent neural network.
            state_in (torch.Tensor): Initial state for the recurrent neural network.
            adv (numpy.ndarray): Advantages obtained from the rollouts.
            prob (torch.Tensor): Probability densities of the selected actions.
            ret (numpy.ndarray): Returns obtained from the rollouts.

        Returns:
            float: Approximated KL divergence between the old and updated policies, based on Schulman et. al(2017).
        """
        p_pi_all = torch.tensor([], dtype=torch.float32, device=device)
        value_out_all = torch.tensor([], dtype=torch.float32, device=device)

        length = rnn_in.shape[0]
        rnn_in = rnn_in.permute(1, 0, 2)

        #Iterate over all timesteps for each pulse
        for i in range(length):
            _, p_pi, value_out, state_out = self.get_action(rnn_in[:, i, :].unsqueeze(1), state_in)
            state_in = state_out.permute(1, 0, 2)
            value_out_all = torch.cat((value_out_all, value_out.unsqueeze(1)), dim=1)
            p_pi_all = torch.cat((p_pi_all, p_pi), dim=0)

        adv_tensor = torch.from_numpy(adv).to(device=device)
        ret_tensor = torch.from_numpy(ret).to(device=device)

        value_out_all = torch.roll(value_out_all, shifts=value_out_all.size(1), dims=1)
        value_out_all = value_out_all.reshape(-1)

        #Policy Function Loss
        ratio = torch.exp(torch.log(p_pi_all) - torch.log(prob))
        surr1 = ratio * adv_tensor
        surr2 = torch.clamp(ratio, 1 - args.eps, 1 + args.eps) * adv_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value Function Loss
        value_loss = (ret_tensor - value_out_all) ** 2
        value_loss = value_loss.mean()

        # Final Loss
        loss = policy_loss + args.w_v * value_loss

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad)

        # Take a step
        self.optimizer.step()

        # Return KL divergence approximation
        approx_kl = -(torch.log(p_pi_all) - torch.log(prob)).mean()

        return approx_kl


def count_parameters(model):
    """
    Counts the total number of trainable parameters in the PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model for which parameters are counted.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

agent=PPO()
print(count_parameters(agent.model))

env = gym.make(args.env)
print(agent.model)

rf_list = torch.empty(1, args.el, 2, device=device)
rew_list = np.array([])


#Training Loop
for it in range(args.max):

    input_list = torch.tensor([], dtype=torch.float32, device=device)   #Store generated RF pulses
    val_list = torch.tensor([], dtype=torch.float32, device=device)     #Store value function estimates
    prob_list = torch.tensor([], dtype=torch.float32, device=device)    #Store probability densities

    rnn_in_ = torch.ones((256,1, 2), dtype=torch.float32, device=device)        #Initial input to RNN is a vector of ones
    state_in_ = torch.zeros((1, 256, 256), dtype=torch.float32, device=device)  #Inital hidden state is a vector of zeros

    #Disable gradient tracking before computing rollouts
    with torch.no_grad():  

        #Inner loop, 1 (A_t,P_t) pair generated per iteration t for each of the 256 RF pulse in the batch
        for ep in range(args.el):

            if ep==0:
                input_list = torch.clone(rnn_in_).permute(1,0,2)
            else:
                input_list = torch.cat((input_list, rnn_in_.permute(1,0,2)), dim=0)

            pi_clip, p_pi, value_out, state_out=agent.get_action(rnn_in_, state_in_)

            if ep == 0:
                val_list = torch.clone(value_out).unsqueeze(1)
            else:
                val_list = torch.cat((val_list, value_out.unsqueeze(1)), dim=1)


            rnn_in_= torch.clone(pi_clip).unsqueeze(1)
            state_in_ = state_out.permute(1,0, 2)
            prob_list = torch.cat((prob_list, p_pi), dim=0)


    input_list = torch.cat((input_list, rnn_in_.permute(1,0,2)), dim=0)
    input_list_tmp = input_list[-32:, :, :].permute(1, 0, 2)

    input_list_tmp[:, :, 0] *= args.amp
    input_list_tmp[:, :, 1] *= args.ph
    input_list_tmp[:, :, 0] = 2 * (input_list_tmp[:, :, 0] / max_rad) - 1
    input_list_tmp[:, :, 1] = (input_list_tmp[:, :, 1] - input_list_tmp[:, 0, 1].unsqueeze(1)) / torch.tensor([math.pi], device=device)

    rf_list = torch.cat((rf_list, input_list_tmp), dim=0)   #Store generated RF pulses at each iteration

    #Feed actions to Bloch simulator and compute rewards
    with torch.no_grad():  

            m = nn.Parameter(input_list_tmp[:, :, 0].to(device))
            p = nn.Parameter(input_list_tmp[:, :, 1].to(device))

            b1 = nn.functional.interpolate(
                torch.stack([m, p], dim=1),
                size=int(args.sampling_rate/2.0),
                mode='linear',
                align_corners=True
            )
            
            b1 = torch.stack([torch.cat((b1[:, 0, :], torch.fliplr(b1[:, 0, :])), dim=1),
                            torch.cat((b1[:, 1, :], torch.fliplr(b1[:, 1, :])), dim=1)], dim=1)
            b1 = torch.cat((ref_pulse, b1), dim=0)

            # Simulation
            t = 0
            done = False
            total_rewards = 0.0
            while not done:
                Mt, rews, done = env.step(b1[..., t])
                t += 1
                total_rewards += rews
            env.reset()

            rew = total_rewards[1:, ...].detach().cpu().numpy()

            # SAR
            amp = ((b1[:, 0] + 1.0) * env.max_amp * 1e+4 / 2).pow(2).sum(-1)  # (G^2)
            sar = amp * env.du / len(env) * 1e6  # (mG^2)*sec
            
            # Magnetization
            Mt1 = Mt[:, :200, 0, :]  # passband
            Mt2 = Mt[:, 200:, 0, :]  # stopband
            ripple1 = torch.max(torch.abs(1. + Mt1[..., 2]), dim=1)[0]
            ripple2 = torch.max(torch.abs(1. - Mt2[..., 2]), dim=1)[0]
            mz = Mt[:, :, 0, 2].detach().cpu().numpy()
            rew_list = np.append(rew_list, rew)


    # Update statistics
    idx = np.argmax(rew)
    info.update('Rew/Mean', np.mean(rew))
    info.update('SAR/Mean', sar.mean().item())
    info.update('Mz1/Mean', ripple1.mean().item())
    info.update('Mz2/Mean', ripple2.mean().item())
    info.update('Rew/Best', rew[idx])
    info.update('SAR/Best', sar[idx].item())
    info.update('Mz1/Best', ripple1[idx].item())
    info.update('Mz2/Best', ripple2[idx].item())
    info.update('Time', time.time() - start_t)   

    
    # Log summary statistics
    if (it + 1) % args.log_step == 0:
        # Inversion profile
        profile = plt.figure(1)
        plt.plot(np.concatenate((env.df[200:200 + 800], env.df[:200], env.df[200 + 800:200 + 1600])),
                    np.concatenate((mz[idx, 200:200 + 800], mz[idx, :200], mz[idx, 200 + 800:200 + 1600])))
        # logger.image_summary(profile, it + 1, 'profile')

        # RF pulse magnitude
        t = np.linspace(0, env.du / len(env), len(env))
        magnitude = b1[:, 0].detach().cpu().numpy()
        fig_m = plt.figure(2)
        plt.plot(t, magnitude[idx])
        plt.ylim(-1, 1)
        # logger.image_summary(fig_m, it + 1, 'magnitude')

        # RF pulse phase
        phase = b1[:, 1].detach().cpu().numpy()
        fig_p = plt.figure(3)
        plt.plot(t, phase[idx])
        plt.ylim(-1, 1)
        # logger.image_summary(fig_p, it + 1, 'phase')

        if (it + 1) % args.save == 0:
            array_dict = {'magnitude': magnitude, 'phase': phase, 'sar': sar, 'rf_list': rf_list[1:, ...],
                            'mz1': ripple1, 'mz2': ripple2, 'rew': rew, 'rew_list': rew_list}
        # else:
        #     array_dict = {'magnitude': magnitude, 'phase': phase, 'sar': sar,
        #                   'mz1': ripple1, 'mz2': ripple2, 'rew': rew}
            logger.savemat('pulse' + str(it + 1), array_dict)

        logger.scalar_summary(info.val, it + 1)

        info.reset()



    #GAE Algorithm calculation
    val_list_arr = val_list.clone().detach().cpu().numpy()

    target = np.roll(val_list_arr, -1, axis=1) * args.gamma
    target[:, -1] = rew
    delta = target - val_list_arr

    adv_list = []
    advs = 0
    for t in range(args.el, 0, -1):
        advs = args.gamma * args.lmbda * advs + delta[:, t-1]
        adv_list.append(advs)
    adv_list.reverse()
    adv_list = np.array(adv_list)
    normalize_rewards = (adv_list - np.mean(adv_list)) / np.std(adv_list)

    norm_reward_resize = normalize_rewards.flatten(order='F')
    ret = np.repeat(rew[:, np.newaxis], args.el, axis=1).flatten(order='F')


    #Pass Over episods + gradient propagation

    for ee in range(args.epochs):

        kl = agent.train(input_list[:-1, :, :], torch.zeros((1, 256, 256), dtype=torch.float32, device=device), norm_reward_resize, prob_list, ret)
        
        #Break update if KL divergence increases too much (Schulman et al. (2017))
        if kl > args.kl:
            print("break")
            break
        else:
            print("optimize")


