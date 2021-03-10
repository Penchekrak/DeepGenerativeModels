import torch

from torch import nn


class Criterion:
    def __call__(self, model: nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, *args, **kwargs):
        pass


class Image2ImageMSELoss(Criterion):
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __repr__(self):
        return repr(self.mse_loss)

    def __call__(self, model: nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, *args, **kwargs):
        outs = model(batch_x)
        loss = self.mse_loss(outs, batch_x)
        return loss, outs


class Image2ImageBCELoss(Criterion):
    def __init__(self, threshold=0.0):
        self.bce_loss = nn.BCELoss()
        self.threshold = threshold

    def __repr__(self):
        return str(repr(self.bce_loss)) + f"with {self.threshold} as threshold"

    def __call__(self, model: nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, *args, **kwargs):
        outs = model(batch_x)
        target = (batch_x > self.threshold).float()
        loss = self.bce_loss(outs, target)
        return loss, outs


class Image2ImageMixedLoss(Criterion):
    def __init__(self, mse_weight=0.5, bce_weight=0.5, bce_threshold=0.5):
        self.bce_loss = nn.BCELoss()
        self.bce_weight = bce_weight
        self.threshold = bce_threshold
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight

    def __call__(self, model: nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, *args, **kwargs):
        outs = model(batch_x)
        mse_loss = self.mse_loss(outs, batch_x)
        target = (batch_x > self.threshold).float()
        bce_loss = self.bce_loss(outs, target)
        return mse_loss * self.mse_weight + bce_loss * self.bce_weight, outs

def l1_loss(x):
    return torch.mean(torch.sum(torch.abs(x), dim=1))


class Image2ImageMixedLossWithLasso(Criterion):
    def __init__(self, mse_weight=0.5, bce_weight=0.5, bce_threshold=0.5, lasso_weight=0.001):
        self.lasso_weight = lasso_weight
        self.bce_loss = nn.BCELoss()
        self.bce_weight = bce_weight
        self.threshold = bce_threshold
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight


    def calculate_sparse_loss(self, model, image):
        loss = 0
        x = image
        for block in model.encoder[:-1]:
            x = block.conv(x)
            loss += l1_loss(x)
            x = block.act(block.norm(x))
        x = model.encoder[-1](x)
        loss += l1_loss(x)

        for block in model.decoder[:-1]:
            x = block.conv(x)
            loss += l1_loss(x)
            x = block.act(block.norm(x))
        x = model.decoder[-1](x)
        loss += l1_loss(x)
        return loss

    def __call__(self, model: nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, *args, **kwargs):
        outs = model(batch_x)
        mse_loss = self.mse_loss(outs, batch_x)
        target = (batch_x > self.threshold).float()
        bce_loss = self.bce_loss(outs, target)
        l1 = self.calculate_sparse_loss(model, batch_x)
        return mse_loss * self.mse_weight + bce_loss * self.bce_weight + self.lasso_weight * l1, outs


class ClassificationCELoss(Criterion):
    def __init__(self):
        self.ce_loss = nn.CrossEntropyLoss()

    def __repr__(self):
        return repr(self.ce_loss)

    def __call__(self, model: nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, *args, **kwargs):
        outs = model(batch_x)
        loss = self.ce_loss(outs, batch_y)
        return loss, outs
