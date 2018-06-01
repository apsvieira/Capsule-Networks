import argparse
import pandas as pd

import torch
import torch.utils.data as data_utils
import torchvision
from torch import nn

from loss_functions import MarginLoss
from models import CapsNet, BaseLine
from utils import split_indices

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--reconstruction', type=bool, default=False)
parser.add_argument('--model', type=str, default='capsnet')
opts = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    model_selector = {
        'capsnet': CapsNet,
        'baseline': BaseLine
    }

    epochs = opts.epochs
    batch_size = opts.batch_size
    num_workers = opts.num_workers
    patience = opts.patience
    reconstruction = opts.reconstruction
    selected = opts.model

    validation_split = 0.2
    dataset = torchvision.datasets.MNIST

    # Only data augmentation for CapsNet is translation of up to 2px
    # As described in Section 5.
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(0, (0.08, 0.08)),
            torchvision.transforms.ToTensor(),
        ]),
        'test': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    }

    # Create data sets
    train_data = dataset('./datasets', train=True, download=True, transform=data_transforms['train'])
    test_data = dataset('./datasets', train=False, download=False, transform=data_transforms['test'])

    # Split training and validation sets
    train_idx, val_idx = split_indices(len(train_data), validation_split)
    train_sampler = data_utils.sampler.SubsetRandomSampler(train_idx)
    val_sampler = data_utils.sampler.SubsetRandomSampler(val_idx)

    # Create data loaders
    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size,
                                         num_workers=num_workers, sampler=train_sampler)
    val_loader = data_utils.DataLoader(train_data, batch_size=batch_size,
                                       num_workers=num_workers, sampler=val_sampler)
    test_loader = data_utils.DataLoader(test_data, batch_size=batch_size,
                                        num_workers=num_workers)

    # Instantiate default network
    model = model_selector[selected](device=device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    class_loss = nn.CrossEntropyLoss()
    margin_loss = MarginLoss(0.9, 0.1, 0.5)
    reconstruction_loss = nn.MSELoss() if reconstruction else None

    if selected == 'capsnet':
        loss_fns = (class_loss, margin_loss)
        loss_history, acc_history = model.train_model(train_loader, epochs, loss_fns, optimizer,
                                                      val_loader, patience, reconstruction_loss)
    elif selected == 'baseline':
        loss_fns = class_loss
        loss_history, acc_history = model.train_model(train_loader, epochs, loss_fns, optimizer,
                                                      val_loader, patience)
    else:
        raise ValueError("Model selected is not supported. Please choose one of {}".format(model_selector.keys()))

    if patience:
        capsnet = torch.load('./caps_best_model.pth')

    evaluation = model.evaluate_model(test_loader, len(test_data))

    df = pd.DataFrame(data={'val_accuracy': acc_history, 'train_loss': loss_history})
    df.to_csv('./metrics.csv')
    with open('./final_loss.txt', 'w') as f:
        f.write(str(evaluation))

    print(evaluation*100)
