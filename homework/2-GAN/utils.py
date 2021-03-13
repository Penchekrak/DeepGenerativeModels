import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.models.resnet import resnet101
from torch.utils.data import DataLoader
from pytorch_lightning.metrics.functional.accuracy import accuracy
import typing as tp


def permute_labels(
        labels: torch.Tensor
):
    indices = torch.randperm(labels.shape[0])
    return labels[indices]


def create_generator_inputs(
        image_batch: torch.Tensor,
        label_batch: torch.Tensor
):
    extra_shape = image_batch.shape[2:]
    spatial_labels = label_batch.unsqueeze(-1).unsqueeze(-1).repeat(extra_shape)
    return torch.cat((image_batch, spatial_labels), dim=1)


def cov(
        x: torch.Tensor,
        x_mean: torch.Tensor = None
) -> torch.Tensor:
    dim = x.shape[-1]
    if x_mean is None:
        x_mean = x.mean(-1)
    x = x - x_mean
    return x @ x.T / (dim - 1)


@torch.no_grad()
def calculate_activation_statistics(
        generator: nn.Module,
        dataloader: DataLoader,
        classifier: nn.Module
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator.eval()
    classifier.eval()
    # device = next(generator.parameters()).device

    real_activations = []
    fake_activations = []
    for batch_x, batch_y in dataloader:
        batch_y = permute_labels(batch_y)
        # batch_x = batch_x.to(device)
        real_activations.append(classifier.get_activations(batch_x))
        fake_activations.append(classifier.get_activations(generator(batch_x, batch_y)))

    real_activations = torch.vstack(real_activations)
    fake_activations = torch.vstack(fake_activations)
    real_activations_mean = torch.mean(real_activations, dim=0)
    fake_activations_mean = torch.mean(fake_activations, dim=0)

    real_activations_cov = cov(real_activations, real_activations_mean)
    fake_activations_cov = cov(fake_activations, fake_activations_mean)
    return real_activations_mean, real_activations_cov, fake_activations_mean, fake_activations_cov


def calculate_frechet_distance(
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: torch.Tensor,
        sigma2: torch.Tensor,
        eps: float = 1e-6
) -> torch.Tensor:
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = torch.atleast_1d(mu1)
    mu2 = torch.atleast_1d(mu2)

    sigma1 = torch.atleast_2d(sigma1)
    sigma2 = torch.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    #  + eps * torch.eye(sigma1.shape[0]).type_as(mu1)
    eigenvals, _ = torch.eig(sigma1.dot(sigma2), eigenvectors=False)
    tr_covmean = torch.sum(torch.sqrt(eigenvals[0]))

    return (diff.dot(diff) + torch.trace(sigma1) +
            torch.trace(sigma2) - 2 * tr_covmean)


@torch.no_grad()
def calculate_fid(
        model: nn.Module,
        dataloader: DataLoader,
        support_model=None
) -> float:
    if support_model is None:
        if not hasattr(calculate_fid, '_support_model'):
            def get_activations(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                x = torch.flatten(x, 1)

                return x

            model = resnet101(pretrained=True, progress=False)
            model.get_activations = get_activations
            setattr(calculate_fid, '_support_model', model)
    else:
        setattr(calculate_fid, '_support_model', support_model)

    m1, s1, m2, s2 = calculate_activation_statistics(model, dataloader, getattr(calculate_fid, '_support_model'))
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value.item()


def calculate_multilabel_accuracy(
        model: nn.Module,
        dataloader: DataLoader,
) -> float:
    predictions = []
    targets = []
    for batch_x, batch_y in dataloader:
        # batch_x = batch_x.to(device)
        predictions.append(model(batch_x))
        targets.append(batch_y)
    predictions = torch.vstack(predictions)
    targets = torch.vstack(targets)
    return accuracy(predictions, targets).item()
