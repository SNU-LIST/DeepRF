from os import path
import sys
sys.path.append(path.join(path.dirname(__file__), '..'))

import os
import numpy as np
import tensorflow as tf
import time
import argparse
from ast import literal_eval as make_tuple
import random
import gym
import torch
import torch.nn as nn
import envs
import matplotlib.pyplot as plt
from utils.logger import Logger
from utils.summary import EvaluationMetrics
from scipy.io import loadmat


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

# random seed
tf.reset_default_graph()
tf.random.set_random_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# load reference pulse
device = 'cuda' if torch.cuda.is_available() else 'cpu'
preset = loadmat('../data/conv_rf/SLR_exc.mat')
ref_pulse = torch.unsqueeze(torch.from_numpy(np.array(preset['result'], dtype=np.float32)), dim=0).to(device)


# %% function definitions
def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


# %% neural network definition
rnn_in = tf.placeholder(tf.float32, [None, None, output_size])
state_in = tf.placeholder(tf.float32, [None, args.hss])  # hidden state in/out size
adv = tf.placeholder(tf.float32, [None, ])  # advantages
prob = tf.placeholder(tf.float32, [None, ])  # history of probabilities
ret_in = tf.placeholder(tf.float32, [None, ])  # returns for training

cell = tf.contrib.rnn.GRUCell(args.hss, reuse=tf.AUTO_REUSE)
state_out, _ = tf.nn.dynamic_rnn(cell, rnn_in, initial_state=state_in, dtype=tf.float32, time_major=True)
state_out = tf.reshape(state_out, [-1, args.hss])

out = mlp(state_out, hidden_sizes=make_tuple(str(args.v_hs)), activation=tf.nn.relu, output_activation=tf.nn.relu)

amp_policy_out = tf.exp(tf.layers.dense(out, units=1))
ph_policy_out = tf.layers.dense(out, units=1)

mean = tf.concat([amp_policy_out, ph_policy_out], 1)
val_out = tf.layers.dense(out, units=1)

log_std = tf.multiply([max_rad * (1 / args.amp), 2 * np.pi * (1 / args.ph)], [args.amp_std, args.ph_std])
std = log_std

r = tf.random_normal(tf.shape(mean))
pi = mean + tf.multiply(r, std)

pi_clip = tf.clip_by_value(pi, [EPS, -np.inf],
                           [max_rad * (1 / args.amp), np.inf])

middle1 = tf.exp(tf.multiply(tf.square(pi_clip - mean),
                             tf.divide(-tf.ones(tf.shape(std)), tf.scalar_mul(2.0, tf.square(std)))))
p_pi_a = tf.reduce_prod(tf.multiply(middle1,
                                    tf.scalar_mul(1 / np.sqrt(np.pi * 2.0), tf.divide(tf.ones(tf.shape(std)),
                                                                                      std))), axis=1)
p_pi = p_pi_a + EPS
ratio = tf.exp(tf.log(p_pi) - tf.log(prob))  # ratio of pi_theta and pi_theta_old

approx_kl = -tf.reduce_mean(tf.log(p_pi) - tf.log(prob))
surr1 = tf.multiply(ratio, adv)
surr2 = tf.multiply(tf.clip_by_value(ratio, 1 - args.eps, 1 + args.eps), adv)
policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
value_loss = tf.reduce_mean(tf.square(tf.squeeze(val_out) - ret_in))
loss = policy_loss + args.w_v * value_loss

# for gradient clipping
params = tf.trainable_variables()
trainer = tf.train.AdamOptimizer(args.lr)
grads_and_var = trainer.compute_gradients(loss, params)
grads, var = zip(*grads_and_var)
grads, grad_norm = tf.clip_by_global_norm(grads, args.grad)
grads_and_var = list(zip(grads, var))
train_opt = trainer.apply_gradients(grads_and_var)

# saver
saver = tf.train.Saver(max_to_keep=100000)

# %% start training

start_t = time.time()
best_SAR = 1e+10
best_RF = 0
best_ind = 0

logger = Logger('AINV', args)
env = gym.make(args.env)

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

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    rf_list = np.empty((1, args.el, 2))
    rew_list = np.array([])

    for it in range(args.max):

        it_start_t = time.time()

        input_list = np.array([])
        adv_list = np.array([])
        prob_list = np.array([])
        val_list = np.array([])

        rnn_in_ = np.ones((1, args.batch, output_size))
        state_in_ = np.zeros((args.batch, args.hss))

        for ep in range(args.el):

            if ep == 0:
                input_list = np.copy(rnn_in_)
            else:
                input_list = np.vstack((input_list, rnn_in_))

            pi__, state_out_, p_pi__, value_out_ = sess.run([pi_clip, state_out, p_pi, val_out],
                                                            feed_dict={rnn_in: rnn_in_, state_in: state_in_})
            if ep == 0:
                val_list = value_out_
            else:
                val_list = np.hstack((val_list, value_out_))

            rnn_in_ = np.expand_dims(pi__, axis=0)  # output to input
            state_in_ = state_out_  # hidden state n-1 to n
            prob_list = np.append(prob_list, p_pi__)  # save action probability

        input_list = np.vstack((input_list, rnn_in_))
        input_list_tmp = np.swapaxes(np.copy(input_list[-args.el:, :, :]), 0, 1)
        input_list_tmp[:, :, 0] = input_list_tmp[:, :, 0] * args.amp
        input_list_tmp[:, :, 1] = input_list_tmp[:, :, 1] * args.ph

        input_list_tmp[:, :, 0] = 2 * (input_list_tmp[:, :, 0] / max_rad) - 1  # -1 ~ 1
        # input_list_tmp[:, :, 1] = (input_list_tmp[:, :, 1] % (2 * np.pi) - np.pi) / np.pi  # -1 ~ 1
        input_list_tmp[:, :, 1] = (input_list_tmp[:, :, 1] - input_list_tmp[:, 0, 1, np.newaxis]) / np.pi

        rf_list = np.vstack((rf_list, input_list_tmp))

        with torch.no_grad():  # for inference only

            m = nn.Parameter(torch.from_numpy(input_list_tmp[:, :, 0]).to(device))
            p = nn.Parameter(torch.from_numpy(input_list_tmp[:, :, 1]).to(device))

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

        # GAE
        # val_list's size: (args.batch, args.el)
        target = np.roll(val_list, -1, axis=1) * args.gamma
        target[:, -1] = rew
        delta = target - val_list

        adv_list = []
        advs = 0
        for t in range(args.el, 0, -1):
            advs = args.gamma * args.lmbda * advs + delta[:, t-1]
            adv_list.append(advs)
        adv_list.reverse()
        adv_list = np.array(adv_list)
        normalize_rewards = (adv_list - np.mean(adv_list)) / np.std(adv_list)

        input_list_resize = np.expand_dims(
            np.reshape(input_list[:-1, :, :], (args.batch * args.el, output_size)), axis=0)
        norm_reward_resize = normalize_rewards.flatten(order='F')
        ret = np.repeat(rew[:, np.newaxis], args.el, axis=1).flatten(order='F')

        for ee in range(args.epochs):
            _, kl, p_pi_, log_std_, policy_loss_, val_loss_, grad_norm_ = sess.run(
                [train_opt, approx_kl, p_pi, log_std, policy_loss, value_loss, grad_norm],
                feed_dict={rnn_in: input_list[:-1, :, :],
                           state_in: np.zeros((args.batch, args.hss)),
                           adv: norm_reward_resize,
                           prob: prob_list,
                           ret_in: ret})

            # early stopping
            if kl > args.kl:
                break
