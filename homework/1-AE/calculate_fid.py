import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
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

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(dataloader, model, classifier):
    classifier.eval()
    model.eval()
    device = next(model.parameters()).device

    # Здесь ожидается что вы пройдете по данным из даталоадера и соберете активации классификатора для реальных и сгенерированных данных
    # После этого посчитаете по ним среднее и ковариацию, по которым посчитаете frechet distance
    # В целом все как в подсчете оригинального FID, но с вашей кастомной моделью классификации
    # note: не забывайте на каком девайсе у вас тензоры 
    # note2: не забывайте делать .detach()
    # YOUR CODE
    real_activations = []
    fake_activations = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Frechet distance calculation"):
            batch_x = batch_x.to(device)
            real_activations.append(classifier.get_activations(batch_x).detach().cpu().numpy())
            fake_activations.append(classifier.get_activations(model(batch_x)).detach().cpu().numpy())

    real_activations = np.vstack(real_activations)
    fake_activations = np.vstack(fake_activations)
    real_activations_mean = np.mean(real_activations, axis=0)
    fake_activations_mean = np.mean(fake_activations, axis=0)

    real_activations_cov = np.cov(real_activations, rowvar=False)
    fake_activations_cov = np.cov(fake_activations, rowvar=False)
    return real_activations_mean, real_activations_cov, fake_activations_mean, fake_activations_cov


@torch.no_grad()
def calculate_fid(dataloader, model, classifier):
    m1, s1, m2, s2 = calculate_activation_statistics(dataloader, model, classifier)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value.item()
