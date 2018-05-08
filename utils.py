import torch
import numpy as np


def squash(tensor):
    '''
    TODO test
    Squash function, defined in [1]. Works as a nonlinearity for CapsNets.
    Input tensor will be of format (bs, units, C, H, W) or (bs, units, C)
    Norm should be computed on the axis representing the number of units.
    params:
        tensor:    torch Variable containing n-dimensional tensor
    output:
        (||tensor||^2 / (1+ ||tensor||^2)) * tensor/||tensor||
    '''
    norm = torch.norm(tensor, p=2, dim=1, keepdim=True)
    sq_norm = norm ** 2  # Avoid computing square twice

    return tensor.div(norm) * sq_norm / (1 + sq_norm)


def split_indices(num_samples, validation_split):
    split_idx = int(validation_split * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split_idx:], indices[:split_idx]
    return train_idx, val_idx
