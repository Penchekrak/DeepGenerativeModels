from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import model as m

import numpy as np
import torchvision

import torch
from torch import nn
from torch.nn import functional as F

image_size = (128, 128)
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

attrs = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young'
index2attr = {i: j for i, j in enumerate(attrs.split())}
attr2index = {j: i for i, j in enumerate(attrs.split())}
my_beloved_attrs = ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male']  # , 'Wearing_Hat', 'Mustache'

my_beloved_indices = [attr2index[i] for i in my_beloved_attrs]


def get_beloved_attrs(labels):
    return labels[my_beloved_indices]


class CelebaDataModule(LightningDataModule):

    def __init__(self, data_dir: str = "celeba", batch_size: int = 10, num_workers: int = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.celeba = torchvision.datasets.CelebA('celeba', target_type='attr', transform=transforms,
                                                  target_transform=get_beloved_attrs, download=False)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.celeba,
                                                                             lengths=[len(self.celeba) - 500,
                                                                                 500])  # torch.utils.data.Subset(self.celeba, [*range(1000)]),

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                                           num_workers=self.num_workers)


celeba = CelebaDataModule()

images, labels = [], []
for i, (image, label) in zip(range(5), torchvision.datasets.CelebA('celeba', target_type='attr', transform=transforms,
                                                                   target_transform=get_beloved_attrs, download=False)):
    images.append(image.unsqueeze(0))
    labels.append(label.unsqueeze(0))
images, labels = torch.cat(images, 0), torch.cat(labels, 0)

# !L
logger = WandbLogger(project='GAN-homework_2-GAN', save_dir=None, log_model=True)
model_checkpointer = ModelCheckpoint(dirpath='wandb', monitor='fid_score', save_weights_only=True)
trainer = Trainer(logger=logger, callbacks=[model_checkpointer], log_every_n_steps=20, gpus=2, accelerator='ddp',
                  plugins=DDPPlugin(find_unused_parameters=False),
                  max_epochs=30)
model = m.VanillaStarGAN(images, labels, index2attr)

trainer.fit(model, datamodule=celeba)
