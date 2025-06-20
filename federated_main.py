#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time
from src.options import args_parser, validate_args
from src.utils import set_random_args
from src.data_utils import dataset_config
from src.server import Server

def run_fed(args):
    # set args dependent on dataset
    dataset_config(args)
    # set random values for a hyperparameter search
    if args.hyperparam_search:
        set_random_args(args)

    validate_args(args)
    server = Server(args)
    return server.start_server()

if __name__ == '__main__':
    if __name__ == '__main__':
        start_time = time.time()
        args = args_parser()

        val_acc, val_loss, test_acc, test_loss = run_fed(args)
        print(f' \n Results after {args.epochs} global rounds of training:')
        print("|---- Validation Accuracy: {:.2f}%".format(100 * val_acc))
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
