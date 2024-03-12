#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', type=bool, default=False, help='enables wandb logging and disables local logfiles')
    parser.add_argument("--wandb_project", type=str, default='', help='specifies wandb project to log to')
    parser.add_argument("--wandb_entity", type=str, default='',
                        help='specifies wandb username to team name where the project resides')
    parser.add_argument("--wandb_run_name", type=str,
                        help="set run name to differentiate runs, if you don't set this wandb will auto generate one")
    parser.add_argument("--offline", type=bool, default=False, help="set wandb to run in offline mode")
    parser.add_argument('--num_workers', type=int, default=1, help="how many subprocesses to use for data loading.")
    parser.add_argument('--epochs', type=int, default=4000, help="number of rounds of training")
    parser.add_argument("--fed_type", type=str, default='fedavg',
                        help="chose federated algorithm. fedavg, fedavgm, fedadam implemented (so far)")
    parser.add_argument('--num_clients', type=int, default=100, help="number of clients: K")
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_iters', type=int, default=None,
                        help="if set, stops training after local_iters mini-batchs of training")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--global_lr', type=float, default=1,
                        help='learning rate for global model, always 1 for FedAvg version')
    parser.add_argument('--client_lr', type=float, default=0.1, help='learning rate for client models')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum, momentum parameter. default is 0.9 ')
    parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1 for FedADAM')
    parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2 for FedADAM')
    parser.add_argument('--adam_eps', type=float, default=0.01,
                        help='ADAM epsilon value (tau in Reddi et. al.), controls degree of adaptivity.')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model name, options: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152')
    parser.add_argument('--width', type=int, default=2, help='model width factor')
    parser.add_argument('--accu_split', type=int, default=None,
                        help='number of groups to split batch into for gradient accumulation when using very large medels')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imgs")
    parser.add_argument('--norm', type=str, default='group_norm', help="batch_norm, group_norm, instance_norm, or None.")
    parser.add_argument('--k', type=int, default=1, help='factor to widen resnet')
    parser.add_argument('--continue_train', type=str, default='', help="path to model to load to continue training")
    parser.add_argument('--hyperparam_search', type=bool, default=False,
                        help="sets random values within a specified range for a hyper parameter search")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help="name of dataset. mnist, fmnist, cifar10, cifar100")
    parser.add_argument('--frac_client_samples', type=float, default=None,
                        help="select a fraction [0, 1] of dataset samples to train on")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--decay', type=int, default=0,
                        help="Use learning rate decay. 1->use 0->don't use. Default = 0.")
    parser.add_argument('--iid', type=int, default=0, help='Default set to non-IID. Set to 1 for IID.')
    parser.add_argument('--dirichlet', type=int, default=1,
                        help='1 uses a dirichlet distribution to create non-iid data, 0 uses shards according to \
                        Mcmahan(2017) et. al. Default = 1.')
    parser.add_argument('--alpha', type=float, default=0.1, help="alpha of dirichlet, value between 0 and infinity\
                        more homogeneous when higher, more heterogeneous when lower")
    parser.add_argument('--print_every', type=int, default=20)
    parser.add_argument('--eval_over_last_hundred', type=int, default=0, help= 'take final eval as average over last '
                                                                               'hundred rounds of training. Useful for '
                                                                               'particularly noisy training.'
                                                                               ' Default is 0, i.e. false')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    return args


def validate_args(args):
    # check number of classes and number of input channels matches dataset
    if args.dataset == 'cifar100' and args.num_classes != 100:
        raise ValueError(
            f'number of classes is set to {args.num_classes}, needs to be 100 for cifar100 dataset'
        )
    if args.dataset in ['cifar10', 'fmnist', 'mnist'] and args.num_classes != 10:
        raise ValueError(
            f'number of classes is set to {args.num_classes}, needs to be 10 for {args.dataset} dataset'
        )
    if args.dataset == 'femnist' and args.num_classes != 62:
        raise ValueError(
            f'number of classes is set to {args.num_classes}, needs to be 62 for {args.dataset} dataset'
        )
    if args.dataset == 'cifar100' or args.dataset == 'cifar10':
        if args.num_channels != 3:
            raise ValueError(
                f'number of input channels is set to {args.num_channels}, needs to be 3 for {args.dataset} dataset'
            )
    if args.dataset == 'mnist' or args.dataset == 'fmnist' or args.dataset == 'femnist':
        if args.num_channels != 1:
            raise ValueError(
                 f'number of input channels is set to {args.num_channels}, needs to be 1 for {args.dataset} dataset'
            )
 
