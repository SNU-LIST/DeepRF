import argparse
import random
import numpy as np
import torch
import torch.nn as nn


def parse_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    parser.add_argument_group("Common configurations")
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default="0")

    parser.add_argument_group("Logger configurations")
    parser.add_argument("--log_level", type=int, default=10)
    parser.add_argument("--log_step", type=float, default=50)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")

    parser.add_argument_group("Training configurations")
    parser.add_argument("--env", type=str, default="Exc-v51")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gam", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--epoch", type=int, default=4)

    args = parser.parse_args()

    # Fix types of arguments
    assert args.env is not None, "environment must be specified"
    assert args.tag is not None, "tag must be specified"
    assert args.gpu is not None, "gpu must be specified"
    if args.tag is None:
        args.tag = args.env.lower()
    else:
        args.tag = (args.env + '_' + args.tag).lower()
    args.log_step = int(args.log_step)

    # Set random seed
    if args.seed == -1:
        random.seed(None)
        args.seed = random.randrange(0, int(1e4))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Set default logging level
    if args.debug:
        args.log_level = 1
    elif args.quiet:
        args.log_level = 30

    return args


def orthogonal_init(module, val):
    targets = [nn.Conv1d, nn.Conv2d, nn.Linear]
    for p in module.modules():
        if any(isinstance(p, t) for t in targets):
            nn.init.orthogonal_(p.weight, val)
            p.bias.data.zero_()
