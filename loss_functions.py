import torch


class MarginLoss:
    """
    TODO add docstring
    """
    def __init__(self, upper_margin=0.9, lower_margin=0.1, lamb=0.5):
        self.upper_margin = upper_margin
        self.lower_margin = lower_margin
        self.lamb = lamb

    def __call__(self, inputs, targets):
        num_classes = inputs.shape[-1]
        upper = torch.max(self.upper_margin - inputs, torch.zeros_like(inputs))
        lower = torch.max(inputs - self.lower_margin, torch.zeros_like(inputs))
        eye = torch.eye(num_classes).to(targets.device)
        oh_targets = eye.index_select(0, targets)

        L = oh_targets * upper.pow(2) + self.lamb * (1 - oh_targets) * lower.pow(2)
        return L.sum()


