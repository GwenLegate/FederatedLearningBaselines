#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import numpy as np
import wandb
import torch
from torch.optim import Adam
from src.client import Client
from src.utils import average_params, get_model, load_past_model, run_summary, wandb_setup, init_run_dir
from src.eval_utils import validation_inference, test_inference, get_validation_ds
from src.client_utils import get_client_labels
from src.data_utils import get_dataset, split_dataset

class FedAdamServer(object):
    def __init__(self, args):
        self.args = args
        self.run_dir = init_run_dir(args, 'CC')
        self.train_loss = []
        self.train_accuracy = []
        self.epoch = 0

    def start_server(self):
        # load dataset and model and optimizer
        train_dataset, validation_dataset, test_dataset = get_dataset(self.args)
        global_model = get_model(self.args)
        optimizer = Adam(global_model.parameters(), lr=self.args.server_lr, betas=(self.args.beta1, self.args.beta2), weight_decay=1e-5, eps=self.args.adam_eps)

        if len(self.args.continue_train) > 0:
            global_model, user_groups = load_past_model(self.args, global_model)
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
                local_model = Client(args=self.args, train_dataset=train_dataset, validation_dataset=validation_dataset,
                                     idx=idx, client_labels=client_labels[idx], all_client_data=user_groups)

                deltas, loss, results = local_model.train_client(model=copy.deepcopy(global_model), global_round=self.epoch)
                local_deltas.append(copy.deepcopy(deltas))
                local_losses.append(copy.deepcopy(loss))

            loss_avg = sum(local_losses) / len(local_losses)
            self.train_loss.append(loss_avg)

            # update global weights
            global_deltas = average_params(local_deltas)
            self._apply_adam_server_update(global_model, optimizer, copy.deepcopy(global_deltas))

            # Test global model inference on validation set after each round use model save criteria
            val_acc, val_loss = validation_inference(self.args, global_model, validation_dataset_global, self.args.num_workers)
            print(f'Epoch {self.epoch} Validation Accuracy {val_acc * 100}% \nValidation Loss {val_loss} '
                  f'\nTraining Loss (average loss of clients evaluated on their own in distribution validation set): {loss_avg}')

            if val_acc > best_acc:
                # save model if it is best acc
                best_acc = copy.deepcopy(val_acc)
                model_path = f'{self.run_dir}/global_model.pt'
                torch.save(global_model.state_dict(), model_path)

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
        if self.args.wandb:
            wandb.log({'val_acc': val_acc,
                       'test_acc': test_acc,
                       })

        return val_acc, val_loss, test_acc, test_loss

    def _apply_adam_server_update(self, model, optimizer, deltas):
        model_weights = copy.deepcopy(model.state_dict())
        # update batch statistics if necessary
        for key in model_weights.keys():
            if 'running' in key or 'batches' in key:
                model_weights[key] = deltas[key]
        model.load_state_dict(model_weights)

        # update gradients with deltas for the round
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.grad = deltas[name]

        optimizer.step(closure=None)
        optimizer.zero_grad(set_to_none=True)






