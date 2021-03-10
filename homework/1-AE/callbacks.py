import tempfile
from abc import abstractmethod
import neptune
import torch

import typing as tp
import tqdm.auto as tqdm
from sklearn.metrics import accuracy_score
import numpy as np

from IPython import display
from ipywidgets import Output

from io import BytesIO

import matplotlib.pyplot as plt


class BaseTrainer(object):
    pass


# on_fit_start = None,

class OnFitStart:
    @abstractmethod
    def __call__(self, trainer: BaseTrainer, config: tp.Dict[str, str], *args, **kwargs):
        pass


class NeptuneOnFitStart(OnFitStart):
    def __init__(self, name, tags=None):
        self.name = name
        self.tags = tags

    def __call__(self, trainer: BaseTrainer, config: tp.Dict[str, tp.Any], *args, **kwargs):
        neptune.create_experiment(name=self.name, params=config, tags=self.tags)
        # for key, value in config.items():
        #     neptune.set_property(key, value)

    # def __init__(self,
    #              project_qualified_name='penchekrak/sandbox',
    #              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZjZkODM3ZjMtYTY4Yy00ZjQ0LThmNTUtNGE0ZTY2M2E0N2RhIn0='):
    #     neptune.init(project_qualified_name, api_token)


# on_epoch_start = None,

class OnEpochStart:
    @abstractmethod
    def __call__(self, trainer: BaseTrainer, epoch: int, *args, **kwargs):
        pass


# on_training_batch_start = None,

class OnTrainingBatchStart:
    @abstractmethod
    def __call__(self, trainer: BaseTrainer, batch_x: torch.Tensor, batch_y: torch.Tensor, *args, **kwargs):
        pass


# on_training_batch_end = None,

class OnTrainingBatchEnd:
    @abstractmethod
    def __call__(self, trainer: BaseTrainer, batch_x: torch.Tensor, batch_y: torch.Tensor,
                 loss: torch.Tensor, *args, **kwargs):
        pass


class NeptuneOnTrainingBatchEnd(OnTrainingBatchEnd):

    def __call__(self, trainer: BaseTrainer, batch_x: torch.Tensor, batch_y: torch.Tensor, loss: torch.Tensor, *args,
                 **kwargs):
        neptune.log_metric('train loss', loss.item())


# on_validation_batch_start = None,


class OnValidationBatchStart:
    @abstractmethod
    def __call__(self, trainer: BaseTrainer, batch_x: torch.Tensor, batch_y: torch.Tensor, *args, **kwargs):
        pass


# on_validation_batch_end = None,


class OnValidationBatchEnd:
    @abstractmethod
    def __call__(self, trainer: BaseTrainer, batch_x: torch.Tensor, batch_y: torch.Tensor, loss: torch.Tensor,
                 metric_results: tp.DefaultDict[str, tp.List[tp.Any]], *args, **kwargs):
        pass


class NeptuneValidationBatchEnd(OnValidationBatchEnd):

    def __call__(self, trainer: BaseTrainer, batch_x: torch.Tensor, batch_y: torch.Tensor, loss: torch.Tensor,
                 metric_results: tp.DefaultDict[str, tp.List[tp.Any]], *args, **kwargs):
        neptune.log_metric('validation loss', loss.item())
        for key, value in metric_results.items():
            neptune.log_metric(key, value[-1])


# on_epoch_end = None,

class OnEpochEnd:
    @abstractmethod
    def __call__(self, trainer: BaseTrainer, epoch: int):
        pass


class MeanAccuracyOnEpochEnd(OnEpochEnd):
    def __call__(self, trainer: BaseTrainer, epoch: int):
        for key, value in trainer.metric_results.items():
            neptune.log_metric(key, epoch, np.mean(value[-1]))


# on_fit_end = None,

class OnFitEnd:
    @abstractmethod
    def __call__(self, trainer: BaseTrainer):
        pass


class ClfAccuracyOnFitEnd(OnFitEnd):
    def __call__(self, trainer: BaseTrainer):
        true_labels = []
        predict_labels = []
        trainer.model.eval()
        with torch.no_grad():
            for image, label in tqdm.tqdm(trainer.val_dataloader, desc='Accuracy calculation', leave=False):
                true_labels.append(label)
                outs = trainer.model(image.to(trainer.device))
                _, predicted = torch.max(outs.data, 1)
                predict_labels.append(predicted)

        true_labels = torch.cat(true_labels, dim=0).numpy()
        predict_labels = torch.cat(predict_labels, dim=0).cpu().numpy()
        accuracy = accuracy_score(true_labels, predict_labels)
        print("Final accuracy:", accuracy)
        neptune.log_metric("Final accuracy", accuracy)


class NeptuneOnFitEnd(OnFitEnd):
    def __call__(self, trainer: BaseTrainer):
        tmp = BytesIO()
        torch.save(trainer.model.state_dict(), tmp)
        tmp.seek(0)
        neptune.log_artifact(tmp, destination='last_model.ckpt')
        neptune.stop()


class PlotLossesOnFitEnd(OnFitEnd):
    def __call__(self, trainer: BaseTrainer):
        print('Losses charts')
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.plot(trainer.train_losses)
        plt.xlabel('step')
        plt.ylabel('train loss')
        plt.subplot(1, 2, 2)
        plt.plot(trainer.val_losses)
        plt.xlabel('step')
        plt.ylabel('val loss')
        plt.show()


class SampleFromModelOnFitEnd(OnFitEnd):
    def __init__(self, samples_grid_shape=(2, 4)):
        self.sample_grid_shape = samples_grid_shape
        self.n_samples = samples_grid_shape[0] * samples_grid_shape[1]
        self.figsize_mult = 4

    def __call__(self, trainer: BaseTrainer):
        print('Samples drawn from model', flush=True)
        samples = trainer.model.sample(self.n_samples)
        figure, axs = plt.subplots(*self.sample_grid_shape, figsize=(
            self.figsize_mult * self.sample_grid_shape[1], self.figsize_mult * self.sample_grid_shape[0]))
        for i, ax in enumerate(axs.flat):
            ax.imshow(samples[i].squeeze().detach().cpu())
        neptune.log_image('samples', figure)
        figure.show()
        plt.show()


class DisplayReconstructionOnFitEnd(OnFitEnd):
    def __init__(self, num_recos=4):
        self.num_recos = num_recos
        self.figsize_mult = 4

    def __call__(self, trainer: BaseTrainer):
        print('Reconstructed images', flush=True)
        batch_x, batch_y = next(iter(trainer.val_dataloader))
        images = batch_x[:self.num_recos].to(trainer.device)
        figure, axs = plt.subplots(2, self.num_recos, figsize=(
            self.figsize_mult * self.num_recos, self.figsize_mult * 2))
        for i in range(self.num_recos):
            axs[0, i].imshow(images[i].squeeze().detach().cpu())
            axs[1, i].imshow(trainer.model(images[i:i + 1]).squeeze().detach().cpu())
        neptune.log_image('reconstructions', figure)
        figure.show()
        plt.show()
