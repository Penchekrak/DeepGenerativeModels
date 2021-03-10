from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import tqdm.auto as tqdm
import typing as tp
import callbacks as cb


class Trainer(cb.BaseTrainer):
    def __init__(
            self,
            on_fit_start: tp.List[cb.OnFitStart] = None,
            on_epoch_start: tp.List[cb.OnEpochStart] = None,
            on_training_batch_start: tp.List[cb.OnTrainingBatchStart] = None,
            on_training_batch_end: tp.List[cb.OnTrainingBatchEnd] = None,
            on_validation_batch_start: tp.List[cb.OnValidationBatchStart] = None,
            on_validation_batch_end: tp.List[cb.OnValidationBatchEnd] = None,
            on_epoch_end: tp.List[cb.OnEpochEnd] = None,
            on_fit_end: tp.List[cb.OnFitEnd] = None,
            **kwargs
    ):
        self.on_fit_start = on_fit_start or []
        self.on_epoch_start = on_epoch_start or []
        self.on_training_batch_start = on_training_batch_start or []
        self.on_training_batch_end = on_training_batch_end or []
        self.on_validation_batch_start = on_validation_batch_start or []
        self.on_validation_batch_end = on_validation_batch_end or []
        self.on_epoch_end = on_epoch_end or []
        self.on_fit_end = on_fit_end or []

    def fit(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            epochs: int,
            metrics=None,
            device='cpu',
            **kwargs
    ):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.metrics = metrics or dict()
        self.epochs = epochs

        self.config = {
            'model': repr(self.model),
            'optimizer': repr(self.optimizer),
            'criterion': repr(self.criterion),
            'metrics': ', '.join(metric_name for metric_name in self.metrics.keys()),
            'epochs': self.epochs,
        }

        self.train_losses = []
        self.val_losses = []

        for func in self.on_fit_start:
            func(self, self.config)
        try:
            with tqdm.tqdm(range(self.epochs), desc="Training epochs", unit="epoch") as epoch_progress_bar:
                for epoch in epoch_progress_bar:
                    for func in self.on_epoch_start:
                        func(self, epoch)
                    # train step
                    self.model.train()
                    with tqdm.tqdm(self.train_dataloader, desc="Train", unit="batch",
                                   leave=False) as train_progress_bar:
                        for batch_idx, (batch_x, batch_y) in enumerate(train_progress_bar):
                            for func in self.on_training_batch_start:
                                func(self, batch_x, batch_y)

                            self.optimizer.zero_grad()
                            loss, out = self.criterion(model, batch_x.to(self.device), batch_y.to(self.device))
                            loss.backward()
                            self.optimizer.step()

                            self.train_losses.append(loss.item())

                            for func in self.on_training_batch_end:
                                func(self, batch_x, batch_y, loss)
                    # validation step
                    self.model.eval()
                    metric_results = defaultdict(list)
                    with tqdm.tqdm(self.val_dataloader, desc='Validation', unit='batch',
                                   leave=False) as validation_progress_bar:
                        for batch_idx, (batch_x, batch_y) in enumerate(validation_progress_bar):
                            for func in self.on_validation_batch_start:
                                func(self, batch_x, batch_y)

                            with torch.no_grad():
                                loss, out = self.criterion(model, batch_x.to(self.device), batch_y.to(self.device))
                                self.val_losses.append(loss.item())

                                for metric_name, metric_func in self.metrics.items():
                                    metric_results[metric_name].append(metric_func(batch_y, out))

                            for func in self.on_validation_batch_end:
                                func(self, batch_x, batch_y, loss, metric_results)

                    for func in self.on_epoch_end:
                        func(self, epoch)
        except KeyboardInterrupt:
            print('Keyboard Interruption, teardown')
        finally:
            for func in self.on_fit_end:
                func(self)

        return model
