import torch
from torch import nn
from collections import defaultdict
import matplotlib.pyplot as plt


class DataPrep:
    def __init__(self, celeba, batch_size=64):
        self.index2attr = {i: j for i, j in enumerate(celeba.attr_names)}
        self.attr2index = {j: i for i, j in enumerate(celeba.attr_names)}

        self.freqs = self.get_freqs(celeba)
        self.sel_attr = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 
                         'Male', 'Smiling']
        self.sel_idxs = [self.attr2index[attr] for attr in self.sel_attr]
        self.sel_index2attr = {i:j for i, j in enumerate(self.sel_attr)}
        self.sel_attr2index = {j:i for i, j in enumerate(self.sel_attr)}
        self.loader = self.get_loader(celeba, batch_size)
    
    def get_labels(self, attr_batch):
        return attr_batch[:, self.sel_idxs]

    def get_bar(self):
        attr_freq = [(self.index2attr[key], value) 
                     for key, value in self.freqs.items()]
        attr_freq.sort(key=lambda x: -x[1])
        x, height = zip(*attr_freq)
        plt.figure(figsize=(20, 12))
        plt.xticks(rotation=75)
        plt.bar(x, height)
        plt.show()
    
    @staticmethod
    def get_loader(celeba, batch_size):
        celeba_dataloader = torch.utils.data.DataLoader(celeba, batch_size, 
                                                        shuffle=True)
        return celeba_dataloader

    @staticmethod
    def get_freqs(celeba):
        freqs = defaultdict(int)
        for column in range(celeba.attr.shape[1]):
            freqs[column] += celeba.attr[:, column].sum()
        freqs = {key: value.item() for key, value in freqs.items()}
        return freqs


def optimizer_to(optim, device):
    # Code from
    # https://github.com/pytorch/pytorch/issues/8741#issuecomment-402129385
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def compute_gradient_penalty(critic: nn.Module, 
                             real_samples: torch.tensor, 
                             fake_samples: torch.tensor):
    device = next(critic.parameters()).device
    epsilon = torch.rand((real_samples.shape[0], 1, 1, 1)).to(device)
    x_hat = epsilon * real_samples + (1 - epsilon) * fake_samples
    x_hat.requires_grad = True
    cr_x_hat, _ = critic(x_hat)
    gradients = torch.autograd.grad(outputs=cr_x_hat, 
                                    inputs=x_hat, 
                                    grad_outputs=torch.ones(
                                        cr_x_hat.shape, 
                                        requires_grad=False, 
                                        device=device),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True,
                                    )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    return torch.mean((gradients.norm(2, dim=1) - 1) ** 2)


def permute_labels(labels: torch.tensor):
    #  перемешиваем батч
    return labels[torch.randperm(labels.shape[0])]
