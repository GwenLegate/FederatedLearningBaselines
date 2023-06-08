from src.options import args_parser
from src.data_utils import get_dataset, dataset_config
from src.utils import get_model, wandb_setup
from src.eval_utils import test_inference
import torch
from torch import nn
from torch.utils.data import DataLoader

args = args_parser()
# set args dependent on dataset
dataset_config(args)

run_dir = './run_data/'
test_nets = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
lr = [0.1, 0.1, 0.1, 0.1, 0.1]
epochs = [100, 100, 100, 100, 100]

# set centralized params
args.norm = 'batch_norm'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')
train_dataset, _, test_dataset = get_dataset(args)
train_dataset, test_dataset = train_dataset, test_dataset
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=args.num_workers, pin_memory=True)
criterion = nn.CrossEntropyLoss().to(device)

for i in range(len(test_nets)):
    print(f'training {test_nets[i]}')
    args.model = test_nets[i]
    args.client_lr = lr[i]
    args.epochs = epochs[i]
    model = get_model(args).to(device)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.client_lr, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.client_lr, weight_decay=1e-4)

    train_loss, train_accuracy = [], []
    epoch = 0
    while epoch < args.epochs:
        print(f'Epoch {epoch}')
        model.train()
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        train_acc, train_loss = test_inference(args, model, train_dataset, args.num_workers)
        test_acc, test_loss = test_inference(args, model, test_dataset, args.num_workers)
        print(f'train_acc: {train_acc}\ntrain_loss: {train_loss}\ntest_acc: {test_acc}\ntest_loss: {test_loss}')
        epoch += 1
