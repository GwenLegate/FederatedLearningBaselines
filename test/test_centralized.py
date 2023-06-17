#!/usr/bin/env python3
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.options import args_parser
from src.data_utils import get_dataset, dataset_config
from src.utils import get_model, wandb_setup
from src.eval_utils import test_inference
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    args = args_parser()
    dataset_config(args)
    # run_dir = './run_data/'
    run_dir = f'/scratch/{os.environ.get("USER", "glegate")}/{args.wandb_run_name}'
    args.norm = 'batch_norm'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(args).to(device)
    print(f'using device: {device}')
    wandb_setup(args, model, run_dir)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.client_lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.client_lr, weight_decay=1e-4)

    # LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64, eta_min=1e-5)

    train_dataset, _, test_dataset = get_dataset(args)
    train_dataset, test_dataset = train_dataset, test_dataset
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss().to(device)
    print(f'training {args.model}')


    train_loss, train_accuracy = [], []
    epoch = 0
    while epoch < args.epochs:
        print(f'Epoch {epoch}')
        model.to(device)
        model.train()
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        train_acc, train_loss = test_inference(args, model, train_dataset, args.num_workers)
        test_acc, test_loss = test_inference(args, model, test_dataset, args.num_workers)
        print(f'train_acc: {train_acc}\ntrain_loss: {train_loss}\ntest_acc: {test_acc}\ntest_loss: {test_loss}')
        wandb.log({'central_train_acc': train_acc,
                   'central_train_loss': train_loss,
                   'central_test_acc': test_acc,
                   'central_test_loss': test_loss}, step=epoch)
        epoch += 1
    print(f'FINAL ACC {args.model}:\n'
        '______________________________________________________________________________________________\n'
        f'train_acc: {train_acc}\ntrain_loss: {train_loss}\ntest_acc: {test_acc}\ntest_loss: {test_loss}\n'
        '______________________________________________________________________________________________\n')
    wandb.log({'final_central_test_acc': test_acc,
               'final_central_test_loss': test_loss})
