import argparse

import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision
from torch import nn

from loss_functions import MarginLoss
from models import CapsNet
from utils import split_indices

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--reconstruction', type=bool, default=False)
opts = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, epochs, loss_fns, opt, validation_loader=None, patience=None, reconstruction=None):
    model.train()
    loss_history = torch.zeros(epochs)
    acc_history = torch.zeros(epochs)
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        loss_sum = 0
        for i, data in enumerate(train_loader):
            print('\rstarting batch #{:5.0f}\r'.format(i))
            input, target = data
            input, target = input.to(device), target.to(device)

            opt.zero_grad()
            log_probs, reconstructed_img = model(input)

            loss = 0
            for loss_fn in loss_fns:
                loss += loss_fn(log_probs, target)

            if reconstruction:
                loss += 0.0005 * reconstruction(reconstructed_img, target)

            loss_sum += loss.item()
            loss.backward()
            opt.step()

        loss_history[epoch] = loss_sum / len(train_loader)
        print('Loss in epoch {}: {}'.format(epoch + 1, loss_history[epoch]))
        torch.save(model, './caps_epoch{}.pth'.format(epoch))
        if patience:
            acc_history[epoch] = evaluate_model(model, validation_loader,
                                                len(validation_loader) * validation_loader.batch_size)
            print('Validation loss in epoch {}: {}'.format(epoch + 1, acc_history[epoch]))
            if acc_history[epoch] > best_val_acc:
                best_val_acc = acc_history[epoch]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print("Early Stopping in epoch {}.".format(epoch))
                    return loss_history, acc_history

    return loss_history, acc_history


def evaluate_model(model, data_loader, num_samples):
    hits = 0.0
    model.eval()

    for i, data in enumerate(data_loader):
        images, targets = data
        with torch.no_grad():
            images = images.to(device)
            targets = targets.to(device)

            log_probs = model(images)
            predictions = F.softmax(log_probs, dim=-1)
            predictions = predictions.max(dim=-1)[1]
            hits += (predictions == targets).sum().item()

    model.train()
    return hits / num_samples


if __name__ == '__main__':

    epochs = opts.n_epochs
    batch_size = opts.batch_size
    num_workers = opts.num_workers
    patience = opts.patience
    reconstruction = opts.reconstruction

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
    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    val_loader = data_utils.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler)
    test_loader = data_utils.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    # Instantiate default network
    capsnet = CapsNet()
    capsnet = capsnet.to(device)

    optimizer = torch.optim.Adam(capsnet.parameters())
    class_loss = nn.CrossEntropyLoss()
    margin_loss = MarginLoss(0.9, 0.1, 0.5)
    reconstruction_loss = nn.MSELoss() if reconstruction else None

    loss_fns = (class_loss, margin_loss)

    loss_history, acc_history = train(capsnet, train_loader, epochs, loss_fns, optimizer,
                                      val_loader, patience, reconstruction_loss)

    evaluation = evaluate_model(capsnet, test_loader, len(test_data))

    print(evaluation*100)

