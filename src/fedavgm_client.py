#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from src.client_utils import DatasetSplit, train_test

class FedAvgMClient(object):
    def __init__(self, args, train_dataset, validation_dataset, idx, client_labels, all_client_data):
        self.args = args
        self.client_idx = idx
        self.lr = self.args.client_lr
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.all_client_data = all_client_data
        self.train_idxs = self.all_client_data[self.client_idx]['train']
        self.validation_idxs = self.all_client_data[self.client_idx]['validation']
        self.trainloader, self.testloader = train_test(self.args, self.train_dataset, self.validation_dataset,
                                                       list(self.train_idxs), list(self.validation_idxs), args.num_workers)
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # set of all labels (training and validation) in the client dataset
        self.labels = client_labels
        self.client_data_proportions = self.data_proportions()

    def get_model(self):
        """
        Returns local model after update
        """
        return self.model_after_training

    def get_client_labels(self, dataset, train_idxs, validation_idxs, num_workers, unique=True):
        """
        Creates a set of all labels present in both train and validation sets of a client dataset
        Args:
            dataset: the complete dataset being used
            train_idxs: the indices of the training samples for the client
            validation_idxs: the indices of the validation samples for the client
            num_workers: how many sub processes to use for data loading
            unique: (bool) if True return the set of client labels, if False returns
        Returns: Set of all labels present in both train and validation sets of a client dataset.
        """
        all_idxs = np.concatenate((train_idxs, validation_idxs), axis=0)
        dataloader = DataLoader(DatasetSplit(dataset, all_idxs), batch_size=len(dataset), shuffle=False,
                        num_workers=num_workers, pin_memory=True)
        _, labels = zip(*[batch for batch in dataloader])

        if unique:
            return labels[0].unique()
        else:
            return labels[0]

    def data_proportions(self):
        client_labels = np.array(self.get_client_labels(self.train_dataset, self.train_idxs, self.validation_idxs,
                                                        self.args.num_workers, False))
        count_labels = client_labels.shape[0]
        count_client_labels = []
        for i in range(self.args.num_classes):
            count_client_labels.append(int(np.argwhere(client_labels == i).shape[0]))
        count_labels = np.array(count_labels)
        count_client_labels = np.array(count_client_labels)

        return count_client_labels / count_labels

    def compute_grad(self, model):
        """
        independently computes the gradient for evaluation purposes for one step without affecting the regular training.
        uses batch size that is the entire client dataset for this step
        Args:
            model: the model to use for the gradient computation
        Returns:
            state dict of gradients
        """
        trainloader = DataLoader(DatasetSplit(self.train_dataset, self.train_idxs),
                                 batch_size=int(len(self.train_idxs)), shuffle=True, num_workers=self.args.num_workers,
                                 pin_memory=True)
        model.train()
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            model.zero_grad()
            logits = model(images)

            if self.args.mask:
                logits = self.weighted_log_softmax(logits)

            loss = self.criterion(logits, labels)
            loss.backward()
            grad_dict = {k:v.grad for k, v in zip(model.state_dict(), model.parameters())}
            return grad_dict

    def train_client(self, model, global_round):
        initial_weights = copy.deepcopy(model.to('cpu')).state_dict()
        # Set mode to train model
        model.to(self.device)
        model.train()
        epoch_loss = []
        # optional variable to return required params as needed
        optional_eval_results = None

        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-4)

        # learning rate decay
        if self.args.decay == 1:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

        # counts number of rounds of sgd
        local_iter_count = 0

        # *** BEGIN TRAINING ***
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                local_iter_count += 1
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()

                logits = model(images)
                loss = self.criterion(logits, labels)

                if self.args.accu_split is not None:
                    if batch_idx % self.args.accu_split == 0:
                        local_iter_count += 1
                        loss.backward()
                        optimizer.step()
                        if self.args.decay == 1:
                            scheduler.step()
                        model.zero_grad()
                        batch_loss.append(loss.item())
                else:
                    local_iter_count += 1
                    loss.backward()
                    optimizer.step()
                    if self.args.decay == 1:
                        scheduler.step()
                    model.zero_grad()
                    batch_loss.append(loss.item())

                if self.args.local_iters is not None:
                    if self.args.local_iters == local_iter_count:
                        break
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        model.to('cpu')
        delta = self._get_delta(copy.deepcopy(model).state_dict(), initial_weights)
        return delta, sum(epoch_loss) / len(epoch_loss), optional_eval_results

    def evaluate_client_model(self, idxs_clients, model):
        """ evaluates an individual client model on the datasets of all other clients selected for this round
        Args:
            idxs_clients: array of indices of the other clients selected for this round
            model: the client model to test
        Returns: A vector of accuracies using the model of client i and the datasets of all clients i and j (where j neq i)
                """
        all_acc = []
        model.eval()

        for idx in idxs_clients:
            _, self.testloader = train_test(self.args, self.train_dataset, self.validation_dataset,
                                            list(self.all_client_data[idx]['train']),
                                            list(self.all_client_data[idx]['validation']), self.args.num_workers)
            acc, _ = self.inference(model)
            all_acc.append(acc)
        return all_acc

    def inference(self, model):
        """
        Returns the inference accuracy and loss.
        """
        model.to(self.args.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        model.to('cpu')
        return accuracy, loss

    def _get_delta(self, params1, params2):
        '''
        Computes delta of model weights or gradients (params2-params1)
        grad update w+ = w - delta_w --> delta_w = w - w+
        Args:
            params1: state dict of weights or gradients
            params2: state dict of weights or gradients
        Returns:
            state dict of the delta between params1 and params2
        '''
        params1 = copy.deepcopy(params1)
        params2 = copy.deepcopy(params2)
        for k, _ in params1.items():
            params2[k] = params2[k].float()
            params2[k] -= params1[k].float()
        return params2
