from os import path
import sys
sys.path.append(path.join(path.dirname(__file__), '..'))

import os
import time
import argparse
import gym
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import envs
import matplotlib.pyplot as plt
from utils.common import parse_args
from utils.logger import Logger
from utils.summary import EvaluationMetrics



def main(args):
    # Select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create logger and environment
    logger = Logger('AINV', args)
    logger.log("Gradient ascent for env '{}'".format(args.env))
    env = gym.make(args.env)

    # Create summary statistics
    info = EvaluationMetrics([
        'Mean/Loss',
        'Mean/SAR',
        'Best/SAR1',
        'Best/SAR2',
        'Best/SAR3',
        'Best/SAR4',
        'Best/SAR5',
        'Best/DIFF1',
        'Best/DIFF2',
        'Best/DIFF3',
        'Best/DIFF4',
        'Best/DIFF5',
        'Best/RIPPLE1',
        'Best/RIPPLE2',
        'Best/RIPPLE3',
        'Best/RIPPLE4',
        'Best/RIPPLE5',
        'Step/Time',
        'Step/Idx1',
        'Step/Idx2',
        'Step/Idx3',
        'Step/Idx4',
        'Step/Idx5',
    ])

    # Create input vectors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preset = loadmat('../data/conv_rf/SLR_exc.mat')
    ref_pulse = torch.unsqueeze(torch.from_numpy(np.array(preset['result'], dtype=np.float32)), dim=0).to(device)

    if args.preset is None:
        # Create random normal vector
        m = torch.FloatTensor(args.samples, args.sampling_rate).uniform_(-1.0, 1.0)
        p = torch.FloatTensor(args.samples, args.sampling_rate).uniform_(-1.0, 1.0)
    else:
        # Load preset and upsample if necessary
        preset = loadmat(args.preset)
        pulse = np.array(preset['result'], dtype=np.float32)
        m = torch.clamp(torch.FloatTensor(pulse[:args.samples, :, 0]), -1.0 + 1e-4, 1.0 - 1e-4)
        p = torch.FloatTensor(pulse[:args.samples, :, 1])

    m = nn.Parameter(torch.atanh(m).to(device))
    p = nn.Parameter(p.to(device))

    # Gradient descent
    optimizer = torch.optim.Adam([m, p], lr=1e-2)
    scheduler = ReduceLROnPlateau(optimizer)
    st = time.time()
    for e in range(args.episodes):
        b1 = torch.stack([torch.tanh(m), p], dim=1)
        b1_ = torch.cat((ref_pulse, b1), dim=0)

        # Simulation
        t = 0
        done = False
        total_rewards = 0.0
        while not done:
            Mt, rews, done = env.step(b1_[..., t])
            t += 1
            total_rewards += rews
        env.reset()

        # Calculate loss from environment
        loss_ = -total_rewards[1:, ...]
        loss = loss_.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # Update statistics
        diff = torch.mean(torch.square(Mt[1:, :1000, 0, 1] - Mt[0, :1000, 0, 1]), dim=1)
        difff = torch.mean(torch.square(Mt[1:, :, 0, 1] - Mt[0, :, 0, 1]), dim=1)
        mxy_ref = torch.sqrt(Mt[0, :, 0, 0] ** 2 + Mt[0, :, 0, 1] ** 2).detach().cpu().numpy()
        mxy = torch.sqrt(Mt[1:, :, 0, 0] ** 2 + Mt[1:, :, 0, 1] ** 2).detach().cpu().numpy()
        pb = torch.sqrt(torch.sum(Mt[1:, :1000, 0, 0], dim=1) ** 2 + torch.sum(Mt[1:, :1000, 0, 1], dim=1) ** 2)
        ripple = torch.max(torch.sqrt(Mt[1:, 1000:, 0, 0] ** 2 + Mt[1:, 1000:, 0, 1] ** 2), dim=1)[0]
        amp = ((b1[:, 0, :] + 1.0) * env.max_amp * 1e4 / 2).pow(2).sum(-1)
        sar = amp * env.du / len(env) * 1e6

        idx1 = 0
        idx2 = 0
        idx3 = 0
        idx4 = 0
        idx5 = 0

        best_SAR1 = 100000
        best_SAR2 = 100000
        best_SAR3 = 100000
        best_SAR4 = 100000
        best_SAR5 = 100000

        for i in range(args.samples):
            if (diff[i]) < 1e-2 and sar[i] < best_SAR1 and ripple[i] < 0.01:
                idx1 = i
                best_SAR1 = sar[i]

        for i in range(args.samples):
            if (diff[i]) < 1e-3 and sar[i] < best_SAR2 and ripple[i] < 0.01:
                idx2 = i
                best_SAR2 = sar[i]

        for i in range(args.samples):
            if (diff[i]) < 1e-4 and sar[i] < best_SAR3 and ripple[i] < 0.01:
                idx3 = i
                best_SAR3 = sar[i]

        for i in range(args.samples):
            if (diff[i]) < 1e-5 and sar[i] < best_SAR4 and ripple[i] < 0.01:
                idx4 = i
                best_SAR4 = sar[i]

        for i in range(args.samples):
            if (diff[i]) < 1e-6 and sar[i] < best_SAR5 and ripple[i] < 0.01:
                idx5 = i
                best_SAR5 = sar[i]

        info.update('Mean/Loss', loss.item())
        info.update('Mean/SAR', sar.mean().item())
        info.update('Best/SAR1', sar[idx1].item())
        info.update('Best/SAR2', sar[idx2].item())
        info.update('Best/SAR3', sar[idx3].item())
        info.update('Best/SAR4', sar[idx4].item())
        info.update('Best/SAR5', sar[idx5].item())
        info.update('Best/DIFF1', diff[idx1].item())
        info.update('Best/DIFF2', diff[idx2].item())
        info.update('Best/DIFF3', diff[idx3].item())
        info.update('Best/DIFF4', diff[idx4].item())
        info.update('Best/DIFF5', diff[idx5].item())
        info.update('Best/RIPPLE1', ripple[idx1].item())
        info.update('Best/RIPPLE2', ripple[idx2].item())
        info.update('Best/RIPPLE3', ripple[idx3].item())
        info.update('Best/RIPPLE4', ripple[idx4].item())
        info.update('Best/RIPPLE5', ripple[idx5].item())
        info.update('Step/Time', time.time() - st)
        info.update('Step/Idx1', idx1)
        info.update('Step/Idx2', idx2)
        info.update('Step/Idx3', idx3)
        info.update('Step/Idx4', idx4)
        info.update('Step/Idx5', idx5)

        # Log summary statistics
        if (e + 1) % args.log_step == 0:
            logger.log('Summary statistics for episode {}'.format(e + 1))

            # Excitation (magnitude) profile
            profile = plt.figure(1)
            plt.plot(np.concatenate((env.df[1000:1000 + 1500], env.df[:1000], env.df[1000 + 1500:1000 + 3000])),
                     np.concatenate((mxy[idx3, 1000:1000 + 1500], mxy[idx3, :1000], mxy[idx3, 1000 + 1500:1000 + 3000])), 'b')
            plt.plot(np.concatenate((env.df[1000:1000 + 1500], env.df[:1000], env.df[1000 + 1500:1000 + 3000])),
                     np.concatenate((mxy_ref[1000:1000 + 1500], mxy_ref[:1000], mxy_ref[1000 + 1500:1000 + 3000])), 'r')
            logger.image_summary(profile, e + 1, 'profile')

            # RF pulse magnitude
            fig_m = plt.figure(2)
            plt.plot(b1[idx3, 0, :].detach().cpu().numpy())
            plt.ylim(-1, 1)
            logger.image_summary(fig_m, e + 1, 'magnitude')

            # RF pulse phase
            fig_p = plt.figure(3)
            plt.plot(b1[idx3, 1, :].detach().cpu().numpy())
            logger.image_summary(fig_p, e + 1, 'phase')

            # matlab save
            array_dict = {'pulse': b1.detach().cpu().numpy(),
                          'sar': sar.detach().cpu().numpy(),
                          'loss_arr': loss_.detach().cpu().numpy(),
                          'ripple': ripple.detach().cpu().numpy(),
                          'difff': difff.detach().cpu().numpy(),
                          'diff': diff.detach().cpu().numpy(),
                          'pb': pb.detach().cpu().numpy()}
            logger.savemat('pulse' + str(e + 1), array_dict)

            logger.scalar_summary(info.val, e + 1)
            info.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="RF Pulse Design using Gradient Ascent"
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=int(1e4))
    parser.add_argument("--sampling_rate", type=int, default=256)
    parser.add_argument("--preset", type=str, default=None)
    args = parse_args(parser)
    main(args)
