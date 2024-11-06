import numpy as np
from src.femnist_dataset import FEMNIST
from src.imagenet32 import Imagenet32
from torchvision import datasets, transforms
from src.sampling import iid_split, noniid_fedavg_split, noniid_dirichlet_equal_split
import ssl


def dataset_config(args):
    '''
    sets dependent args based on selected dataset if the required args are different from the default
    Args:
        args: set of args passed to arg parser
    '''
    if args.dataset == 'cifar100':
        args.num_classes = 100
    if args.dataset == 'femnist':
        args.num_classes = 62
        args.num_channels = 1
    if args.dataset == 'imagenet32':
        args.num_classes = 1000
    if args.dataset == 'mnist':
        args.num_channels = 1
    if args.dataset == 'flowers102':
        args.num_classes = 102
        args.image_size = 224


def get_num_samples_per_label(self, dataset_labels):
    labels = np.array(dataset_labels)
    examples_per_label = []
    for i in range(self.args.num_classes):
        examples_per_label.append(int(np.argwhere(labels == i).shape[0]))
    return examples_per_label


def get_dataset(args):
    """
    Returns train, validation and test datasets
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        data_dir = '../data/'

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(f'{data_dir}{args.dataset}/', train=True, download=True,
                                             transform=transform_train)
            validation_dataset = datasets.CIFAR10(f'{data_dir}{args.dataset}/', train=True, download=False,
                                                  transform=transform_test)
            test_dataset = datasets.CIFAR10(f'{data_dir}{args.dataset}/', train=False, download=False,
                                            transform=transform_test)

        if args.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(f'{data_dir}{args.dataset}/', train=True, download=True,
                                              transform=transform_train)
            validation_dataset = datasets.CIFAR100(f'{data_dir}{args.dataset}/', train=True, download=False,
                                                   transform=transform_test)
            test_dataset = datasets.CIFAR100(f'{data_dir}{args.dataset}/', train=False, download=False,
                                             transform=transform_test)

    elif args.dataset == 'femnist':
        data_dir = '../data/femnist/'
        train_dataset = FEMNIST(root=data_dir, train=True, download=True)
        mean = train_dataset.train_data.float().mean()
        std = train_dataset.train_data.float().std()

        apply_transform = transforms.Compose([
            transforms.RandomCrop(24, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_dataset = FEMNIST(data_dir, train=True, download=False, transform=apply_transform)
        validation_dataset = FEMNIST(data_dir, train=True, download=False, transform=test_transform)
        test_dataset = FEMNIST(data_dir, train=False, download=False, transform=test_transform)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
            validation_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=apply_transform)
        else:
            data_dir = '../data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=False, transform=apply_transform)
            validation_dataset = datasets.FashionMNIST(data_dir, train=True, download=False, transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=False, transform=apply_transform)
    elif args.dataset == 'imagenet32':
        data_dir = '../data/imagenet32/'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = Imagenet32(
            f'{data_dir}out_data_train/',
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        validation_dataset = Imagenet32(
            f'{data_dir}out_data_train/',
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        test_dataset = Imagenet32(
            f'{data_dir}out_data_val/',
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset == 'flowers102':
        data_dir = '../data/flowers/'
        train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                               ])
        test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ])
        train_dataset = datasets.Flowers102(f'{data_dir}train/', split='train', transform=train_transforms, download=True)
        validation_dataset = datasets.Flowers102(f'{data_dir}train/', split='train', transform=test_transforms, download=True)
        test_dataset = datasets.Flowers102(f'{data_dir}test/', split='test', transform=test_transforms, download=True)

    else:
        print(f'dataset {args.dataset} is invalid')
        exit(1)

    return train_dataset, validation_dataset, test_dataset


def split_dataset(train_dataset, args):
    ''' Splits the dataset between args.num_clients clients and further partitions each clients subset into training
        and validation sets
    Args:
        train_dataset: the complete training dataset
        args: the user specified options for the run
    Returns:
        user_groups: a nested dict where keys are the user index and values are another dict with keys
        'train' and 'validation' which contain the corresponding sample indices for training and validation subsets of
        each clients partition
    '''
    if args.iid:
        # Sample IID user data
        user_groups = iid_split(train_dataset, args.num_clients)
    else:
        if args.dirichlet:
            print(f'Creating non iid client datasets using Dirichlet distribution')
            user_groups = noniid_dirichlet_equal_split(train_dataset, args.alpha, args.num_clients, args.num_classes)
            if args.frac_client_samples:
                user_groups = noniid_dirichlet_equal_split(train_dataset, args.alpha, args.num_clients,
                                                           args.num_classes, data_subset=args.frac_client_samples)
        else:
            print(f'Creating non iid client datasets using shards')
            user_groups = noniid_fedavg_split(train_dataset, args.num_clients, client_shards=2)

    return user_groups
