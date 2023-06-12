import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data_utils import get_dataset
from src.models import ResNet18, ResNet34
from src.fed_avg_client import DatasetSplit

def get_validation_ds(num_clients, user_groups, validation_dataset):
    # combine indicies for validation sets of each client to test global model on complete set
    for i in range(num_clients):
        if i == 0:
            idxs_val = user_groups[i]['validation']
        else:
            idxs_val = np.concatenate((idxs_val, user_groups[i]['validation']), axis=0)

    return DatasetSplit(validation_dataset, idxs_val)

def validation_inference(args, model, validation_dataset, num_workers):
    """
    Returns the validation accuracy and loss.
    """
    model.to(args.device)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    valloader = DataLoader(validation_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)

    for batch_idx, (images, labels) in enumerate(valloader):
        images, labels = images.to(args.device), labels.to(args.device)

        # Inference
        try:
            outputs = model(images)
        except:
            outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss = loss / (batch_idx + 1)
    model.to('cpu')
    return accuracy, loss

def test_inference(args, model, test_dataset, num_workers):
    """
    Returns the test accuracy and loss.
    """
    model.to(args.device)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(args.device), labels.to(args.device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss = loss/batch_idx
    model.to('cpu')
    return accuracy, loss

def test_model(args, model_path):
    model = ResNet18(args=args)
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    _, _, test_dataset = get_dataset(args)
    test_acc, _ = test_inference(args, model, test_dataset, args.num_workers)
    print(test_acc)
    return test_acc
