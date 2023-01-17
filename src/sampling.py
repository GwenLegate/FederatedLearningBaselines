#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
from collections import Counter
import numpy as np

def set_non_iid_params(len_dataset, num_users):
    """
    Set num_shards and num_imgs parameters interactively for non iid cases (fixes the problem of capping users at 100)
        Args:
            len_dataset (Int):    Length of the dataset being used.
            num_users (Int):      Number of clientd in federated iid setting

        Returns:
            num_shards (Int):     optimal number of shards for number of users and dataset size
            num_imgs (Int):       optimal number of images for number of users and dataset size
    """
    num_shards = num_users * 2
    num_images = math.floor(len_dataset / num_shards)
    return num_shards, num_images


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    labels = dataset.train_labels.numpy()
    num_shards, num_imgs = set_non_iid_params(labels.shape[0], num_users)
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    #num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
        # create 90/10 train/validation split for client i
        samples = np.array(dict_users[i])
        train_idxs = samples[:int(samples.shape[0] * 0.9)].astype('int64').squeeze()
        validation_idxs = samples[int(samples.shape[0] * 0.1):].astype('int64').squeeze()

        dict_users[i] = {}
        dict_users[i]['train'] = train_idxs
        dict_users[i]['validation'] = validation_idxs
        # remove assigned idxs
        all_idxs = [idx for idx in all_idxs if idx not in samples]
    return dict_users

def shard_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from a given dataset
    :param dataset: the dataset to split iid
    :param num_users: the number of clients to divide the samples between
    :return: dict of training and validation indices for each client
    """

    labels = np.array(dataset.targets)
    num_shards, num_imgs = set_non_iid_params(labels.shape[0], num_users)
    idx_shard = [i for i in range(num_shards)]

    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(labels.shape[0])

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        # create 90/10 train/validation split for client i
        samples = dict_users[i]
        train_idxs = samples[:int(samples.shape[0] * 0.9)].astype('int64').squeeze()
        validation_idxs = samples[int(samples.shape[0] * 0.9):].astype('int64').squeeze()

        dict_users[i] = {}
        dict_users[i]['train'] = train_idxs
        dict_users[i]['validation'] = validation_idxs

    return dict_users

# adapted from https://github.com/google-research/federated/blob/master/utils/datasets/cifar10_dataset.py
# gives equally sized datasets for each client
def equal_class_size_noniid_dirichlet(dataset, alpha, num_clients, num_classes):
    """Construct a federated dataset from the centralized CIFAR-10.
    Sampling based on Dirichlet distribution over categories, following the paper
    Measuring the Effects of Non-Identical Data Distribution for
    Federated Visual Classification (https://arxiv.org/abs/1909.06335).
    Args:
        dataset: The dataset to split
        alpha: Parameter of Dirichlet distribution. Each client
        samples from this Dirichlet to get a multinomial distribution over
        classes. It controls the data heterogeneity of clients. If approaches 0,
        then each client only have data from a single category label. If
        approaches infinity, then the client distribution will approach IID
        partitioning.
        num_clients: The number of clients the examples are going to be partitioned on.
        num_classes: The number of unique classes in the dataset
    Returns:
        a dict where keys are client numbers from 0 to num_clients and nested dict inside of each key has keys train
        and validation containing arrays of the indicies of each sample.
        """
    labels = np.array(dataset.targets)
    dict_users = {}
    multinomial_vals = []
    examples_per_label = []
    for i in range(num_classes):
        examples_per_label.append(int(np.argwhere(labels == i).shape[0]))

    # Each client has a multinomial distribution over classes drawn from a Dirichlet.
    for i in range(num_clients):
        proportion = np.random.dirichlet(alpha * np.ones(num_classes))
        multinomial_vals.append(proportion)

    multinomial_vals = np.array(multinomial_vals)
    example_indices = []

    for k in range(num_classes):
        label_k = np.where(labels == k)[0]
        np.random.shuffle(label_k)
        example_indices.append(label_k)

    example_indices = np.array(example_indices)

    client_samples = [[] for _ in range(num_clients)]
    count = np.zeros(num_classes).astype(int)

    examples_per_client = int(labels.shape[0] / num_clients)

    for k in range(num_clients):
        for i in range(examples_per_client):
            sampled_label = np.argwhere(np.random.multinomial(1, multinomial_vals[k, :]) == 1)[0][0]
            client_samples[k].append(example_indices[sampled_label, count[sampled_label]])
            count[sampled_label] += 1
            if count[sampled_label] == examples_per_label[sampled_label]:
                multinomial_vals[:, sampled_label] = 0
                multinomial_vals = (
                        multinomial_vals /
                        multinomial_vals.sum(axis=1)[:, None])
    for i in range(num_clients):
        np.random.shuffle(np.array(client_samples[i]))
        # create 90/10 train/validation split for each client
        samples = np.array(client_samples[i])
        train_idxs = samples[:int(samples.shape[0] * 0.9)].astype('int64').squeeze()
        validation_idxs = samples[int(samples.shape[0] * 0.9):].astype('int64').squeeze()

        dict_users[i] = {}
        dict_users[i]['train'] = train_idxs
        dict_users[i]['validation'] = validation_idxs

    return dict_users

def unequal_class_size_noniid_dirichlet(dataset, alpha, num_clients, num_classes):
    """Construct a federated dataset from the centralized CIFAR-10.
    Sampling based on Dirichlet distribution over categories, following the paper
    Measuring the Effects of Non-Identical Data Distribution for
    Federated Visual Classification (https://arxiv.org/abs/1909.06335).
    Args:
        dataset: The dataset to split
        alpha: Parameter of Dirichlet distribution. Each client
        samples from this Dirichlet to get a multinomial distribution over
        classes. It controls the data heterogeneity of clients. If approaches 0,
        then each client only have data from a single category label. If
        approaches infinity, then the client distribution will approach IID
        partitioning.
        num_clients: The number of clients the examples are going to be partitioned on.
        num_classes: The number of unique classes in the dataset
    Returns:
        a dict where keys are client numbers from 0 to num_clients and nested dict inside of each key has keys train
        and validation containing arrays of the indicies of each sample.
        """
    labels = np.array(dataset.targets)
    dict_users = {}
    multinomial_vals = []
    examples_per_label = []
    for i in range(num_classes):
        examples_per_label.append(int(np.argwhere(labels == i).shape[0]))

    # Each client has a multinomial distribution over classes drawn from a Dirichlet.
    for i in range(num_clients):
        proportion = np.random.dirichlet(alpha * np.ones(num_classes))
        multinomial_vals.append(proportion)

    multinomial_vals = np.array(multinomial_vals)
    example_indices = []

    for k in range(num_classes):
        label_k = np.where(labels == k)[0]
        np.random.shuffle(label_k)
        example_indices.append(label_k)

    example_indices = np.array(example_indices)

    client_samples = [[] for _ in range(num_clients)]
    count = np.zeros(num_classes).astype(int)

    examples_per_client = int(labels.shape[0] / num_clients)

    for k in range(num_clients):
        for i in range(examples_per_client):
            sampled_label = np.argwhere(np.random.multinomial(1, multinomial_vals[k, :]) == 1)[0][0]
            label_indices = example_indices[sampled_label]
            client_samples[k].append(label_indices[count[sampled_label]])
            count[sampled_label] += 1
            if count[sampled_label] == examples_per_label[sampled_label]:
                multinomial_vals[:, sampled_label] = 0
                multinomial_vals = (
                        multinomial_vals /
                        multinomial_vals.sum(axis=1)[:, None])
    for i in range(num_clients):
        np.random.shuffle(np.array(client_samples[i]))
        # create 90/10 train/validation split for each client
        samples = np.array(client_samples[i])
        train_idxs = samples[:int(samples.shape[0] * 0.9)].astype('int64').squeeze()
        validation_idxs = samples[int(samples.shape[0] * 0.9):].astype('int64').squeeze()

        dict_users[i] = {}
        dict_users[i]['train'] = train_idxs
        dict_users[i]['validation'] = validation_idxs

    return dict_users

def femnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from Leaf's femnist dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # create 90/10 train/validation split for client i
        samples = np.array(list(dict_users[i]))
        train_idxs = samples[:int(samples.shape[0] * 0.9)].astype('int64').squeeze()
        validation_idxs = samples[int(samples.shape[0] * 0.9):].astype('int64').squeeze()

        dict_users[i] = {}
        dict_users[i]['train'] = train_idxs
        dict_users[i]['validation'] = validation_idxs
        # remove assigned idxs
        all_idxs = list((Counter(all_idxs) - Counter(list(dict_users[i]))).elements())
    return dict_users
