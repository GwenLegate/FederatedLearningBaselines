#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import numpy as np
import wandb
import torch
from src.fed_avg_client import FedAvgClient
from src.utils import average_weights, get_model, load_past_model, run_summary, wandb_setup, zero_last_hundred, ncm
from src.eval_utils import validation_inference, test_inference, get_validation_ds
from src.client_utils import get_client_labels
from src.data_utils import get_dataset, split_dataset

class FedAvgServer(object):
    def __init__(self, args):
        self.args = args

    def start_server(self):
        # create dir to save run artifacts
        #run_dir = f'/scratch/{os.environ.get("USER", "glegate")}/{self.args.wandb_run_name}'
        run_dir = './run_data/'
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir, mode=0o755, exist_ok=True)

        # set lists for last 100 item average
        last_hundred_test_loss, last_hundred_test_acc, last_hundred_val_loss, last_hundred_val_acc = zero_last_hundred()
        # load dataset
        train_dataset, validation_dataset, test_dataset = get_dataset(self.args)

        # init server model
        global_model = get_model(self.args)
        # if this run is a continuation of training for a failed run, load previous model and client distributions
        if len(self.args.continue_train) > 0:
            global_model, user_groups = load_past_model(self.args, global_model)
        else:
            # splits dataset among clients
            user_groups = split_dataset(train_dataset, self.args)
            # save the user_groups dictionary for later access
            user_groups_to_save = f'{run_dir}/user_groups.pt'
            torch.save(user_groups, user_groups_to_save)

        # list of set of labels present for each client
        client_labels = get_client_labels(train_dataset, user_groups, self.args.num_workers, self.args.num_classes)
        # get validation ds by combining indicies for validation sets of each client
        validation_dataset_global = get_validation_ds(self.args.num_clients, user_groups, validation_dataset)

        # ncm init if using ncm
        if self.args.ncm:
            global_model, prev_class_means = ncm(self.args, global_model, train_dataset, user_groups, [i for i in range(self.args.num_classes)])

        # set best acc to update saved global model
        val_acc, _ = validation_inference(self.args, global_model, validation_dataset_global,
                                              self.args.num_workers)
        best_acc = copy.deepcopy(val_acc)

        # set up wandb connection
        if self.args.wandb:
            wandb_setup(self.args, global_model, run_dir)

        run_summary(self.args)

        train_loss, train_accuracy = [], []
        epoch = 0
        # **** TRAINING LOOPS STARTS HERE ****
        while epoch < self.args.epochs:
            local_losses = []
            local_weights = []
            if self.args.ncm:
                class_count = torch.zeros(self.args.num_classes).to(self.args.device)
                class_sums = torch.zeros((self.args.num_classes, 512)).to(self.args.device)

            global_round = f'\n | Global Training Round : {epoch + 1} |\n'
            print(global_round)

            m = max(int(self.args.frac * self.args.num_clients), 1)
            idxs_clients = np.random.choice(range(self.args.num_clients), m, replace=False)

            # get proportions of client labels seen at each round (for ncm eval purposes)
            if self.args.ncm:
                label_prpos = get_client_labels(train_dataset, user_groups, self.args.num_workers, self.args.num_classes,
                                                proportions=True, subset_idxs=idxs_clients)
                print(f'label proportions for round: {label_prpos}')

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

            if self.args.ncm:
                class_count += results[0]
                class_sums += results[1]

                # replace last layer with NCM using data from participating clients
                class_means = torch.div(class_sums, torch.reshape(class_count, (-1, 1)))

                # replace any nans from dividing by zero class samples with previous class means
                for i, c in enumerate(class_count):
                    if c == 0:
                        class_means[i] = prev_class_means[i]

                prev_class_means = class_means
                global_model.linear.weight.data = torch.nn.functional.normalize(class_means)

            # Test global model inference on validation set after each round use model save criteria
            val_acc, val_loss = validation_inference(self.args, global_model, validation_dataset_global, self.args.num_workers)
            print(f'Epoch {epoch} Validation Accuracy {val_acc * 100}% \nValidation Loss {val_loss} '
                  f'\nTraining Loss (average loss of clients evaluated on their own in distribution validation set): {loss_avg}')

            if val_acc > best_acc:
                # save model if it is best acc
                best_acc = copy.deepcopy(val_acc)
                model_path = f'{run_dir}/global_model.pt'
                torch.save(global_model.state_dict(), model_path)

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

        # model_path = f'/scratch/{os.environ.get("USER", "glegate")}/{self.args.wandb_run_name}/global_model.pt'
        model_path = f'{run_dir}/global_model.pt'
        # load best model for testing
        global_model.load_state_dict(torch.load(model_path))
        '''if self.args.ncm:
            global_model = ncm(self.args, global_model, train_dataset, user_groups)'''

        # Test inference after completion of training
        test_acc, test_loss = test_inference(self.args, global_model, test_dataset, self.args.num_workers)

        # last 100 avg acc and loss
        last_hundred_test_loss = sum(last_hundred_test_loss) / len(last_hundred_test_loss)
        last_hundred_test_acc = sum(last_hundred_test_acc) / len(last_hundred_test_acc)
        last_hundred_val_loss = sum(last_hundred_val_loss) / len(last_hundred_val_loss)
        last_hundred_val_acc = sum(last_hundred_val_acc) / len(last_hundred_val_acc)

        if self.args.wandb:
            wandb.log({'val_acc': val_acc,
                       'test_acc': test_acc,
                       'last_100_val_acc': last_hundred_val_acc,
                       'last_100_test_acc': last_hundred_test_acc
                       })

        return val_acc, val_loss, test_acc, test_loss, last_hundred_val_acc, last_hundred_val_loss, \
               last_hundred_test_acc, last_hundred_test_loss






