#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import numpy as np
import wandb
import torch
from src.fedavgm_client import FedAvgMClient
from src.utils import (average_params, get_model, load_past_model, run_summary, wandb_setup, zero_last_hundred,
                       last_hundred_update, last_hundred_avg, init_run_dir)
from src.eval_utils import validation_inference, test_inference, get_validation_ds
from src.client_utils import get_client_labels
from src.data_utils import get_dataset, split_dataset

class FedAvgMServer(object):
    def __init__(self, args):
        self.args = args
        self.run_dir = init_run_dir(args, 'CC')
        self.train_loss = []
        self.train_accuracy = []
        self.epoch = 0

    def start_server(self):
        if self.args.eval_over_last_hundred:
            last_hundred = zero_last_hundred()

        # load dataset and init model and momentum
        train_dataset, validation_dataset, test_dataset = get_dataset(self.args)
        global_model = get_model(self.args)
        global_weights = global_model.state_dict()
        server_momentum = copy.deepcopy(global_weights)
        for k, v in server_momentum.items():
            server_momentum[k] = torch.zeros_like(v)

        if len(self.args.continue_train) > 0:
            global_model, user_groups, momentum = load_past_model(self.args, global_model, server_momentum)
        else:
            user_groups = split_dataset(train_dataset, self.args)
            user_groups_to_save = f'{self.run_dir}/user_groups.pt'
            torch.save(user_groups, user_groups_to_save)

        # get validation ds by combining indicies for validation sets of each client
        client_labels = get_client_labels(train_dataset, user_groups, self.args.num_workers, self.args.num_classes)
        validation_dataset_global = get_validation_ds(self.args.num_clients, user_groups, validation_dataset)

        # init best acc obtained for model
        val_acc, _ = validation_inference(self.args, global_model, validation_dataset_global, self.args.num_workers)
        best_acc = copy.deepcopy(val_acc)

        # set up wandb connection
        if self.args.wandb:
            wandb_setup(self.args, global_model, self.run_dir)

        run_summary(self.args)

        # **** TRAINING LOOPS STARTS HERE ****
        while self.epoch < self.args.epochs:
            local_losses = []
            local_deltas = []

            global_round = f'\n | Global Training Round : {self.epoch + 1} |\n'
            print(global_round)

            m = max(int(self.args.frac * self.args.num_clients), 1)
            idxs_clients = np.random.choice(range(self.args.num_clients), m, replace=False)

            # for each selected client, init model weights with global weights and train lcl model for local_ep epochs
            for idx in idxs_clients:
                local_model = FedAvgMClient(args=self.args, train_dataset=train_dataset, validation_dataset=validation_dataset,
                                          idx=idx, client_labels=client_labels[idx], all_client_data=user_groups)

                delta, loss, results = local_model.train_client(model=copy.deepcopy(global_model), global_round=self.epoch)
                local_deltas.append(copy.deepcopy(delta))
                local_losses.append(copy.deepcopy(loss))

            loss_avg = sum(local_losses) / len(local_losses)
            self.train_loss.append(loss_avg)

            # update global weights
            global_deltas = average_params(local_deltas)
            server_momentum, global_weights = self._apply_momentum_server_update(copy.deepcopy(server_momentum),
                                                                                 copy.deepcopy(global_weights),
                                                                                 copy.deepcopy(global_deltas))
            global_model.load_state_dict(global_weights)

            # Test global model inference on validation set after each round use model save criteria
            val_acc, val_loss = validation_inference(self.args, global_model, validation_dataset_global, self.args.num_workers)
            print(f'Epoch {self.epoch} Validation Accuracy {val_acc * 100}% \nValidation Loss {val_loss} '
                  f'\nTraining Loss (average loss of clients evaluated on their own in distribution validation set): {loss_avg}')

            if val_acc > best_acc:
                # save model, momentum update if it is best acc
                best_acc = copy.deepcopy(val_acc)
                model_path = f'{self.run_dir}/global_model.pt'
                momentum_path = f'{self.run_dir}/server_momentum.pt'
                torch.save(global_model.state_dict(), model_path)
                torch.save(server_momentum, momentum_path)

            if self.args.eval_over_last_hundred and self.args.epochs - (self.epoch + 1) <= 100:
                test_acc, test_loss = test_inference(self.args, global_model, test_dataset, self.args.num_workers)
                last_hundred = last_hundred_update(last_hundred, (val_loss, val_acc, test_loss, test_acc))

            # print global training loss after every 'i' rounds
            if (self.epoch + 1) % self.args.print_every == 0:
                if self.args.wandb:
                    wandb.log({f'val_acc': val_acc,
                               f'val_loss': val_loss,
                               f'train_loss': loss_avg
                               }, step=self.epoch)

            self.epoch += 1

        # load best model for testing
        model_path = f'{self.run_dir}/global_model.pt'
        global_model.load_state_dict(torch.load(model_path))

        # Test inference after completion of training
        test_acc, test_loss = test_inference(self.args, global_model, test_dataset, self.args.num_workers)
        if self.args.eval_over_last_hundred:
            last_hundred_val_loss, last_hundred_val_acc, last_hundred_test_loss, last_hundred_test_acc = last_hundred_avg(
                self.args, last_hundred, val_acc, test_acc)
            return val_acc, val_loss, test_acc, test_loss, last_hundred_val_acc, last_hundred_val_loss, \
                last_hundred_test_acc, last_hundred_test_loss
        else:
            if self.args.wandb:
                wandb.log({'val_acc': val_acc,
                           'test_acc': test_acc,
                           })

            return val_acc, val_loss, test_acc, test_loss

    def _apply_momentum_server_update(self, momentum, global_weights, global_deltas):
        momentum_update = copy.deepcopy(momentum)

        for k, v in global_weights.items():
            if "running" in k or "batches_tracked" in k:
                pass
            else:
                momentum_update[k] = (self.args.momentum * momentum[k]) + global_deltas[k]
                global_weights[k] -= self.args.global_lr * momentum_update[k]
        return momentum_update, global_weights
