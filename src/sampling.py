#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
from collections import Counter
import numpy as np

def dirichlet_client_sizes(num_units, num_clients, beta, min_units=10):
    """
    Draws an unequal allocation of num_units units (samples, or shards for the McMahan split) across clients.

    This is the quantity skew of NIID-Bench, Federated Learning on Non-IID Data Silos (Li et. al., ICDE 2022,
    https://arxiv.org/abs/2102.02079): sample q ~ Dir_N(beta) and give client j a q_j proportion of the total.
    beta controls the imbalance, with small beta heavily skewed and large beta approaching equal sizes.

    It differs from the reference implementation in how the min_units floor is enforced. NIID-Bench rejects and
    redraws until every client clears the floor, which is fine for the ~10 parties it evaluates but effectively
    never succeeds at the client counts used here: at 450 clients over CIFAR-10, a Dir(0.5) draw clears a floor
    of 10 with probability ~0, so the loop cannot terminate. Instead every client is given min_units up front and
    only the remaining units are shared out by the Dirichlet. That always terminates and hits the floor exactly,
    at the cost of compressing the skew slightly: the marginal is a floor-shifted Dirichlet rather than one
    truncated below. The two agree closely whenever the floor is small next to the mean client size.

    Args:
        num_units: total number of units to allocate across the clients
        num_clients: the number of clients to allocate units to
        beta: concentration parameter of the Dirichlet, must be > 0
        min_units: every client is guaranteed at least this many units
    Returns:
        an integer array of length num_clients, summing to exactly num_units, with every entry >= min_units
    """
    if beta <= 0:
        raise ValueError(f'quantity skew beta must be > 0, got {beta}')
    if min_units < 0:
        raise ValueError(f'min_units must be >= 0, got {min_units}')
    if num_clients * min_units > num_units:
        raise ValueError(f'cannot give each of {num_clients} clients at least {min_units} units out of only '
                         f'{num_units} units in total')

    # hand out the floor first, then let the Dirichlet share out what is left
    free_units = num_units - num_clients * min_units
    proportions = np.random.dirichlet(np.repeat(beta, num_clients))
    proportions = proportions / proportions.sum()
    cuts = (np.cumsum(proportions) * free_units).astype(int)[:-1]
    sizes = np.diff(np.concatenate(([0], cuts, [free_units]))) + min_units
    return sizes.astype(int)

def _train_validation_split(samples, validation_frac=0.1):
    """
    Splits one client's sample indices into disjoint train and validation sets.
    Kept in one place so the three partition strategies cannot drift apart.
    """
    # reshape rather than squeeze: squeeze turns a single-element client into a 0-d array, which is not iterable
    samples = np.asarray(samples).reshape(-1).astype('int64')
    split = int(samples.shape[0] * (1 - validation_frac))
    return samples[:split], samples[split:]

def iid_split(dataset, num_users, client_sizes=None):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :param client_sizes: optional per-client sample counts. Defaults to equally sized clients. Combined with
        dirichlet_client_sizes this is the 'iid-diff-quantity' partition of NIID-Bench.
    :return: dict of image index
    """
    dict_users = {}
    # a single permutation gives disjoint, randomly assigned client partitions
    all_idxs = np.random.permutation(len(dataset))
    if client_sizes is None:
        client_sizes = np.full(num_users, int(len(dataset) / num_users), dtype=int)
    offsets = np.concatenate(([0], np.cumsum(client_sizes))).astype(int)

    for i in range(num_users):
        samples = all_idxs[offsets[i]:offsets[i + 1]]
        # create 90/10 train/validation split for client i
        train_idxs, validation_idxs = _train_validation_split(samples)

        dict_users[i] = {}
        dict_users[i]['train'] = train_idxs
        dict_users[i]['validation'] = validation_idxs
    return dict_users

def noniid_fedavg_split(dataset, num_users, client_shards=2, client_shard_counts=None):
    """
    Sample non-I.I.D client data from a given dataset according to strategy in Communication-Efficient Learning of Deep
    Networks from Decentralized Data (McMahan et. al.). Method generalized for any number of shards per client while the
    implementation in McMahan et. al. is for client_shards=2

    :param dataset: the dataset to split iid
    :param num_users: the number of clients to divide the samples between
    :param client_shards: the number of shards assigned to each client
    :param client_shard_counts: optional per-client shard counts, which must sum to num_users * client_shards.
        Defaults to client_shards for every client. Because shards are a fixed size, allocating an unequal number
        of them is how quantity skew is applied to this strategy, so client sizes are quantised to a multiple of
        the shard size rather than following the Dirichlet exactly.
    :return: dict of training and validation indices for each client
    """

    labels = np.array(dataset.targets)
    num_shards = num_users * client_shards
    num_imgs = math.floor(labels.shape[0] / num_shards)
    if num_imgs < 1:
        raise ValueError(f'the number of images per shard is < 1 ({labels.shape[0]} samples over {num_shards} '
                         f'shards), please select a smaller number of client_shards')
    idx_shard = [i for i in range(num_shards)]

    if client_shard_counts is None:
        client_shard_counts = np.full(num_users, client_shards, dtype=int)
    elif int(np.sum(client_shard_counts)) != num_shards:
        raise ValueError(f'client_shard_counts must sum to {num_shards}, got {int(np.sum(client_shard_counts))}')

    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(labels.shape[0])

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, int(client_shard_counts[i]), replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        # create 90/10 train/validation split for client i
        train_idxs, validation_idxs = _train_validation_split(dict_users[i])

        dict_users[i] = {}
        dict_users[i]['train'] = train_idxs
        dict_users[i]['validation'] = validation_idxs

    return dict_users

# adapted from https://github.com/google-research/federated/blob/master/utils/datasets/cifar10_dataset.py
# gives equally sized datasets for each client unless client_sizes is supplied

def noniid_dirichlet_equal_split(dataset, alpha, num_clients, num_classes, data_subset=None, client_sizes=None):
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
        data_subset: optionally keep only this fraction of each client's samples
        client_sizes: optional per-client sample counts, which must not sum to more than the dataset size.
        Defaults to equally sized clients, which is what Hsu et. al. specify. Supplying sizes drawn from a
        Dirichlet layers quantity skew on top of the label skew controlled by alpha; the two interact, so the
        result is no longer the partition described in the paper.
    Returns:
        a dict where keys are client numbers from 0 to num_clients and nested dict inside of each key has keys train
        and validation containing arrays of the indicies of each sample.
        """
    try:
        labels = np.array(dataset.targets)
    except AttributeError:
        labels = np.array(dataset._labels)
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

    example_indices = np.array(example_indices, dtype=object)

    client_samples = [[] for _ in range(num_clients)]
    count = np.zeros(num_classes).astype(int)

    if client_sizes is None:
        client_sizes = np.full(num_clients, int(labels.shape[0] / num_clients), dtype=int)
    elif int(np.sum(client_sizes)) > labels.shape[0]:
        raise ValueError(f'client_sizes sum to {int(np.sum(client_sizes))}, more than the {labels.shape[0]} '
                         f'samples in the dataset')

    for k in range(num_clients):
        for i in range(int(client_sizes[k])):
            # every class this client can still draw from has been exhausted by earlier clients
            if multinomial_vals[k, :].sum() <= 0:
                break
            sampled_label = np.argwhere(np.random.multinomial(1, multinomial_vals[k, :]) == 1)[0][0]
            label_indices = example_indices[sampled_label]
            client_samples[k].append(label_indices[count[sampled_label]])
            count[sampled_label] += 1
            if count[sampled_label] == examples_per_label[sampled_label]:
                multinomial_vals[:, sampled_label] = 0
                # a row of all zeros means that client has run out of classes to draw from. Leave it at zero
                # instead of dividing by zero and turning the whole row into NaNs
                row_sums = multinomial_vals.sum(axis=1)[:, None]
                multinomial_vals = np.divide(multinomial_vals, row_sums,
                                             out=np.zeros_like(multinomial_vals), where=row_sums > 0)
    for i in range(num_clients):
        # create 90/10 train/validation split for each client
        samples = np.array(client_samples[i])
        np.random.shuffle(samples)
        if data_subset is not None:
            samples = samples[:int(samples.shape[0] * data_subset)]
        train_idxs, validation_idxs = _train_validation_split(samples)

        dict_users[i] = {}
        dict_users[i]['train'] = train_idxs
        dict_users[i]['validation'] = validation_idxs

    return dict_users