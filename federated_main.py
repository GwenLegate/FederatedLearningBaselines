#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time
from src.options import args_parser, validate_args
from src.utils import set_random_args
from src.data_utils import dataset_config
from src.fedavg_server import FedAvgServer
from src.fedavgm_server import FedAvgMServer
from src.fedadam_server import FedAdamServer

def run_fed(args, fed_type):
    if fed_type == 'fedavg':
        server = FedAvgServer(args)
    elif fed_type == 'fedavgm':
        server = FedAvgMServer(args)
    elif fed_type == 'fedadam':
        server = FedAdamServer(args)
    else:
        raise ValueError(f'type {fed_type} not implemented')
    return server.start_server()

if __name__ == '__main__':
    if __name__ == '__main__':
        start_time = time.time()
        args = args_parser()
        # set args dependent on dataset
        dataset_config(args)

        # set random values for a hyperparameter search
        if args.hyperparam_search:
            set_random_args(args)

        validate_args(args)
        if args.eval_over_last_hundred:
            val_acc, val_loss, test_acc, test_loss, last_hundred_val_acc, last_hundred_val_loss, last_hundred_test_acc,\
            last_hundred_test_loss = run_fed(args, args.fed_type)
            print(f' \n Results after {args.epochs} global rounds of training:')
            print("|---- Validation Accuracy: {:.2f}%".format(100 * val_acc))
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
            print("|---- Last 100 Validation Accuracy: {:.2f}%".format(100 * last_hundred_val_acc))
            print("|---- Last 100 Test Accuracy: {:.2f}%".format(100 * last_hundred_test_acc))
            print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
        else:
            val_acc, val_loss, test_acc, test_loss = run_fed(args, args.fed_type)
            print(f' \n Results after {args.epochs} global rounds of training:')
            print("|---- Validation Accuracy: {:.2f}%".format(100 * val_acc))
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
            print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
