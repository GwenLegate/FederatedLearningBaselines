import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import warnings

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    warnings.filterwarnings("ignore")
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

def wsm(logits):
    """
    computes softmax weighted by class proportions from the client
    Args:
        logits: logits for the mini batch under consideration
    Returns:
        softmax weighted by class proportion
    """
    alphas = torch.from_numpy(self.client_data_proportions).to(self.device)
    log_alphas = alphas.log().clamp_(min=-1e9)
    deno = torch.logsumexp(log_alphas + logits, dim=-1, keepdim=True)
    return log_alphas + logits - deno

def get_client_labels(dataset, user_groups, num_workers, num_classes, proportions=False):
    """
    Creates a List containing the set of all labels present in both train and validation sets for each client,
    optionally returns this list of present lables or a List of proportions of each class in the dataset
    Args:
        dataset: the complete dataset being used
        user_groups: dict of indices assigned to each client
        num_workers: how many sub processes to use for data loading
        num_classes: number of classes in the dataset
        proportions: boolean indicating if class proportions should be returned instead of client labels
    Returns: if proportions is False: List containing the set of all labels present in both train and validation sets
    of each client dataset, indexed by client number. If proportions is True: a list containing the proportion of each
    label of each client dataset, indexed by client number.
    """
    def get_labels(client_idxs):
        dataloader = DataLoader(DatasetSplit(dataset, client_idxs), batch_size=len(dataset), shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        _, labels = zip(*[batch for batch in dataloader])

        if proportions:
            labels = np.asarray(labels[0])
            count_labels = labels.shape[0]
            count_client_labels = []
            for i in range(num_classes):
                count_client_labels.append(int(np.argwhere(labels == i).shape[0]))
            count_client_labels = np.array(count_client_labels)
            return np.unique(labels), count_client_labels / count_labels

        return labels[0].unique()

    if proportions:
        client_groups = user_groups.items()
        client_labels = []
        client_proportions = []
        for client in client_groups:
            unique_labels, label_proportions = get_labels(np.concatenate((client[1]['train'], client[1]['validation']), axis=0))
            client_labels.append(unique_labels)
            client_proportions.append(label_proportions)
        return client_proportions
    else:
        client_groups = user_groups.items()
        client_labels = [get_labels(np.concatenate((client[1]['train'], client[1]['validation']), axis=0))
                        for client in client_groups]
        return client_labels

def train_test(args, train_dataset, validation_dataset, train_idxs, validation_idxs, num_workers):
        """
        Create train and validation dataloaders for a client given train and validation dataset and indices.
        Args:
            train_dataset: the training dataset
            validation_dataset: the validation dataset
            train_idxs: the list of indices of samples used for training for the client
            validation_idxs: the list of indices used for validation for the client
            num_workers: how many processes to use for data loading
        Return:
            train and validation dataloaders for a client
        """
        if args.accu_split is not None:
            trainloader = DataLoader(DatasetSplit(train_dataset, train_idxs),
                                     batch_size=int(args.local_bs/args.accu_split), shuffle=True, num_workers=num_workers, pin_memory=True)
            validationloader = DataLoader(DatasetSplit(validation_dataset, validation_idxs),
                                          batch_size=int(args.local_bs/args.accu_split), shuffle=False,
                                          num_workers=num_workers, pin_memory=True)
        else:
            trainloader = DataLoader(DatasetSplit(train_dataset, train_idxs),
                                 batch_size=args.local_bs, shuffle=True, num_workers=num_workers, pin_memory=True)
            validationloader = DataLoader(DatasetSplit(validation_dataset, validation_idxs),
                                batch_size=args.local_bs, shuffle=False, num_workers=num_workers, pin_memory=True)
        return trainloader, validationloader

