#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import numpy as np
import wandb
import torch
from src.fed_avg_client import FedAvgClient
from src.utils import average_weights, get_model, load_past_model, run_summary, wandb_setup, zero_last_hundred
from src.eval_utils import validation_inference, test_inference, get_validation_ds
from src.client_utils import get_client_labels
from src.data_utils import get_dataset, split_dataset

class FedAvgServer(object):
    def __init__(self, args):
        self.args = args

    def start_server(self):
        # create dir to save run artifacts
        run_dir = f'/scratch/{os.environ.get("USER", "glegate")}/{self.args.wandb_run_name}'
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir, mode=0o755, exist_ok=True)

        # set lists for last 100 item average
        last_hundred_test_loss, last_hundred_test_acc, last_hundred_val_loss, last_hundred_val_acc = zero_last_hundred()
        # load dataset
        train_dataset, validation_dataset, test_dataset = get_dataset(self.args)
        # splits dataset among clients
        user_groups = split_dataset(train_dataset, self.args)
        # save the user_groups dictionary for later access
        user_groups_to_save = f'/scratch/{os.environ.get("USER", "glegate")}/{self.args.wandb_run_name}/user_groups.pt'
        torch.save(user_groups, user_groups_to_save)
        # list of set of labels present for each client
        client_labels = get_client_labels(train_dataset, user_groups, self.args.num_workers, self.args.num_classes)
        # get validation ds by combining indicies for validation sets of each client
        validation_dataset_global = get_validation_ds(self.args.num_clients, user_groups, validation_dataset)
        # init server model
        global_model = get_model(self.args)
        # if this run is a continuation of training for a failed run, load previous model and client distributions
        if len(self.args.continue_train) > 0:
            global_model, user_groups = load_past_model(self.args, global_model)

        # Set the model to training mode and send it to device.
        global_model.to(self.args.device)
        global_model.train()
        print(global_model)

        # set up wandb connection
        if self.args.wandb:
            wandb_setup(self.args, global_model)

        run_summary(self.args)

        train_loss, train_accuracy = [], []
        epoch = 0
        # **** TRAINING LOOPS STARTS HERE ****
        while epoch < self.args.epochs:
            local_losses = []
            local_weights = []

            global_round = f'\n | Global Training Round : {epoch + 1} |\n'
            print(global_round)

            global_model.train()
            m = max(int(self.args.frac * self.args.num_clients), 1)
            idxs_clients = np.random.choice(range(self.args.num_clients), m, replace=False)

            # for each selected client, init model weights with global weights and train lcl model for local_ep epochs
            for idx in idxs_clients:
                local_model = FedAvgClient(args=self.args, train_dataset=train_dataset, validation_dataset=validation_dataset,
                                          idx=idx, client_labels=client_labels[idx], all_client_data=user_groups)

                w, loss, results = local_model.train_client(model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # update global weights with the average of the obtained local weights
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)

            if epoch % 50 == 0:
                # save model as a backup every 50 epochs
                model_path = f'/scratch/{os.environ.get("USER", "glegate")}/{self.args.wandb_run_name}/global_model.pt'
                torch.save(global_model.state_dict(), model_path)

            # Test global model inference on validation set after each round use model save criteria
            val_acc, val_loss = validation_inference(self.args, global_model, validation_dataset_global, self.args.num_workers)
            print(f'Epoch {epoch} Validation Accuracy {val_acc * 100}% \nValidation Loss {val_loss} '
                  f'\nTraining Loss (average loss of clients evaluated on their own in distribution validation set): {loss_avg}')

            if self.args.epochs - (epoch + 1) <= 100:
                last_hundred_val_loss.append(val_loss)
                last_hundred_val_acc.append(val_acc)
                test_acc, test_loss = test_inference(self.args, global_model, test_dataset, self.args.num_workers)
                last_hundred_test_loss.append(test_loss)
                last_hundred_test_acc.append(test_acc)

            # print global training loss after every 'i' rounds
            if (epoch + 1) % self.args.print_every == 0:
                if self.args.wandb:
                    wandb.log({f'val_acc': val_acc,
                               f'val_loss': val_loss,
                               f'train_loss': loss_avg
                               # f'global model test accuarcy': test_acc,
                               # f'global model test loss': test_loss
                               }, step=epoch)

            epoch += 1

        # save final model after training
        model_path = f'/scratch/{os.environ.get("USER", "glegate")}/{self.args.wandb_run_name}/global_model.pt'
        torch.save(global_model.state_dict(), model_path)

        # Test inference after completion of training
        test_acc, test_loss = test_inference(self.args, global_model, test_dataset, self.args.num_workers)

        # last 100 avg acc and loss
        last_hundred_test_loss = sum(last_hundred_test_loss) / len(last_hundred_test_loss)
        last_hundred_test_acc = sum(last_hundred_test_acc) / len(last_hundred_test_acc)
        last_hundred_val_loss = sum(last_hundred_val_loss) / len(last_hundred_val_loss)
        last_hundred_val_acc = sum(last_hundred_val_acc) / len(last_hundred_val_acc)

        if self.args.wandb:
            wandb.log({f'val_acc': val_acc,
                       f'test_acc': test_acc,
                       f'last_100_val_acc': last_hundred_val_acc,
                       f'last_100_test_acc': last_hundred_test_acc
                       })

        return val_acc, val_loss, test_acc, test_loss, last_hundred_val_acc, last_hundred_val_loss, \
               last_hundred_test_acc, last_hundred_test_loss






