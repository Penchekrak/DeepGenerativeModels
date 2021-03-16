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
        real_activations: torch.Tensor,
        fake_activations: torch.Tensor,
        # classifier: nn.Module
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # device = next(generator.parameters()).device
    # classifier.to(real_images[0].device)
    # real_activations = []
    # fake_activations = []
    # for real_image_batch, fake_image_batch in zip(real_images, fake_images):
    #     real_activations.append(classifier.get_activations(real_image_batch))
    #     fake_activations.append(classifier.get_activations(fake_image_batch))
    #
    # real_activations = torch.vstack(real_activations)
    # fake_activations = torch.vstack(fake_activations)
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

    diff = mu1 - mu2

    #  + eps * torch.eye(sigma1.shape[0]).type_as(mu1)
    eigenvals, _ = torch.eig(sigma1 @ sigma2, eigenvectors=False)
    tr_covmean = torch.sum(torch.sqrt(torch.abs(eigenvals[:, 0])))

    return (diff.dot(diff) + torch.trace(sigma1) +
            torch.trace(sigma2) - 2 * tr_covmean)


class FidScore(pl.metrics.Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False, classifier: tp.Optional[nn.Module] = None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.classifier = classifier
        if self.classifier is None:
            import types
            def get_activations(self_, x):
                x = self_.conv1(x)
                x = self_.bn1(x)
                x = self_.relu(x)
                x = self_.maxpool(x)

                x = self_.layer1(x)
                x = self_.layer2(x)
                x = self_.layer3(x)
                x = self_.layer4(x)

                x = self_.avgpool(x)
                x = torch.flatten(x, 1)

                return x

            model = resnet101(pretrained=True, progress=False)
            model.get_activations = types.MethodType(get_activations, model)
            self.classifier = model

        self.add_state("real_activations", default=[], dist_reduce_fx=None)
        self.add_state("fake_activations", default=[], dist_reduce_fx=None)

    @torch.no_grad()
    def update(self, real_images, fake_images) -> None:
        self.real_activations.append(self.classifier.get_activations(real_images))
        self.fake_activations.append(self.classifier.get_activations(fake_images))

    def compute(self):
        m1, s1, m2, s2 = calculate_activation_statistics(torch.cat(self.real_activations, dim=0), torch.cat(self.fake_activations, dim=0))
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value

# @torch.no_grad()
# def calculate_fid(
#         real_images: tp.List[torch.Tensor],
#         fake_images: tp.List[torch.Tensor],
#         support_model: tp.Optional[nn.Module] = None
# ) -> float:
#     if support_model is None:
#         if not hasattr(calculate_fid, '_support_model'):
#             setattr(calculate_fid, '_support_model', model)
#     else:
#         setattr(calculate_fid, '_support_model', support_model)
#
#     m1, s1, m2, s2 = calculate_activation_statistics(real_images, fake_images, getattr(calculate_fid, '_support_model'))
#     fid_value = calculate_frechet_distance(m1, s1, m2, s2)
#
#     return fid_value.item()


# def calculate_multilabel_accuracy(
#         model: nn.Module,
#         dataloader: DataLoader,
# ) -> float:
#     predictions = []
#     targets = []
#     for batch_x, batch_y in dataloader:
#         # batch_x = batch_x.to(device)
#         predictions.append(model(batch_x))
#         targets.append(batch_y)
#     predictions = torch.vstack(predictions)
#     targets = torch.vstack(targets)
#     return accuracy(predictions, targets).item()
