#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
from time import time
import random
import wandb
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import multiprocessing as mp
from src.resnets import ResNet34, ResNet18, ResNet50, ResNet101, ResNet152
from src.vit import ViTForClassification
from torchvision.models import swin_b

def init_run_dir(args, my_env=None):
    """
    Creates a directory to store run artifacts and throws an error if the directory already exists.
    Args:
        args: command line args
        my_env: specifies compute environment so directory is made in the correct location (Mila, Compute Canada or local)
    Returns:
        the path to the run directory
    """
    if my_env == 'Mila':
        run_dir = f'/network/scratch/g/{os.environ.get("USER", "gwendolyne.legate")}/{args.wandb_run_name}'
    elif my_env == 'CC':
        run_dir = f'/scratch/{os.environ.get("USER", "glegate")}/{args.wandb_run_name}'
    else:
        run_dir = './run_data/'

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir, mode=0o755, exist_ok=True)
    else:
        if args.wandb_run_name is not None and not args.continue_train:
            raise Exception(f'Directory {args.wandb_run_name} already exists')

    return run_dir


def average_params(w):
    """
    Returns the average of the local weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_grads(local_grads):
    grad_avg = copy.deepcopy(local_grads[0])
    for i in range(1, len(local_grads)):
        for k, _ in grad_avg.items():
            grad_avg[k] += local_grads[i][k]
    for k, _ in grad_avg.items():
        try:
            grad_avg[k] *= 1 / len(local_grads)
        except RuntimeError:
            grad_avg[k] *= int(1 / len(local_grads))

    return grad_avg

   # set random values for learning rates, local epochs, local batch size for hyperparameter search
def set_random_args(args):
    lrs = [7E-2, 5E-2, 3E-2, 1E-2, 7E-3, 5E-3, 3E-3, 1E-3, 7E-4]
    args.local_bs = random.randrange(5, 120, 5)  # sets a local batch size between 5 and 125 (intervals of 5)
    args.server_lr = 1
    idx = random.randrange(9)
    args.client_lr = lrs[idx]
    args.local_ep = random.randrange(4, 26)
    if idx < 4:
        args.epochs = random.randrange(1000, 3001, 250)
    else:
        args.epochs = random.randrange(2500, 4501, 250)

# method to empirically find a good number of workers for data loaders
def find_optimal_num_workers(training_data, bs):
    for num_workers in range(2, mp.cpu_count()):
        train_loader = DataLoader(training_data,shuffle=True,num_workers=num_workers,batch_size=bs,pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for _, _ in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

def run_summary(args):
    print(f'Run Parameters:\n'
          f'\twandb name: {args.wandb_run_name}\n'
          f'\tlr: {args.client_lr}\n'
          f'\tiid: {args.iid}\n'
          f'\tclients: {args.num_clients}\n'
          f'\tfraction of clients selected: {args.frac}\n'
          f'\tlocal bs: {args.local_bs}\n'
          f'\tlocal epochs: {args.local_ep}\n'
          f'\tlocal iterations: {args.local_iters}\n'
          f'\talpha: {args.alpha}\n'
          f'\tnorm: {args.norm}\n'
          f'\tdataset: {args.dataset}')

def wandb_setup(args, model, run_dir, central=False):
    if args.wandb_run_name:
        os.environ['WANDB_NAME'] = args.wandb_run_name
        os.environ['WANDB_START_METHOD'] = "thread"
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    # need to set wandb run_dir to something I can access to avoid permission denied error.
    # See https://github.com/wandb/client/issues/3437
    # wandb_path = f'/scratch/{os.environ.get("USER","glegate")}/wandb'
    wandb_path = f'{run_dir}/wandb'
    if not os.path.isdir(wandb_path):
        os.makedirs(wandb_path, mode=0o755, exist_ok=True,)

    # if using wandb check project and entity are set
    assert not args.wandb_project == '' and not args.wandb_entity == ''
    wandb.init(dir=wandb_path, project=args.wandb_project, entity=args.wandb_entity)
    if central:
        general_args = {
            "client learning_rate": args.client_lr,
            "epochs": args.epochs,
            "dataset": args.dataset,
            "model": args.model,
            "norm": args.norm,
        }
        pass
    else:
        general_args = {
            "client learning_rate": args.client_lr,
            "epochs": args.epochs,
            "dataset": args.dataset,
            "model": args.model,
            "iid": args.iid,
            "clients": args.num_clients,
            "fraction of clients (C)": args.frac,
            "local epochs (E)": args.local_ep,
            "local iters": args.local_iters,
            "local batch size (B)": args.local_bs,
            "dirichlet": args.dirichlet,
            "alpha": args.alpha,
            "norm": args.norm,
            "decay": args.decay,
        }
    wandb.config.update(general_args)

    # log every 'print_every' epochs

    wandb.watch(model, log_freq=args.print_every)

def get_model(args):
    # BUILD MODEL
    if args.model == 'resnet18':
        return ResNet18(args=args)
    elif args.model == 'resnet34':
        return ResNet34(args=args)
    elif args.model == 'resnet50':
        return ResNet50(args=args)
    elif args.model == 'resnet101':
        return ResNet101(args=args)
    elif args.model == 'resnet152':
        return ResNet152(args=args)
    elif args.model == 'vit':
        return ViTForClassification(args)
    elif args.model == 'swin':
        return swin_b()
    else:
        exit('Error: unrecognized model')

def load_past_model(args, model):
    # if this run is a continuation of training for a failed run, load previous model and client distributions (and momentum for fedavgm)
    model.load_state_dict(torch.load(args.continue_train))
    user_groups_path = f"{args.continue_train.rsplit('/', 1)[0]}/user_groups.pt"
    user_groups = torch.load(user_groups_path)
    return model, user_groups

def get_delta(params1, params2):
    '''
    Computes delta of model weights or gradients (params2-params1)
    Args:
        params1: state dict of weights or gradients
        params2: state dict of weights or gradients
    Returns:
        state dict of the delta between params1 and params2
    '''
    params1 = copy.deepcopy(params1)
    params2 = copy.deepcopy(params2)
    for k, _ in params1.items():
        params2[k] -= params1[k]
    return params2

def compute_accuracy(model, dataloader, device):
    """
    compute accuracy method from NIID-Bench, kept for consistancy with scaffold implementation
    """
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if was_training:
        model.train()

    return correct/float(total)
