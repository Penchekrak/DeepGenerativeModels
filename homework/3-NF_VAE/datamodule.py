import torchvision
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule
import typing as tp
from torchvision import transforms as T


def get_preprocessing_transforms(transforms):
    transforms_list = []
    for transform_name, transform_args in transforms.items():
        transform = getattr(T, transform_name, None)
        assert (transform is not None), f"torchvision.transforms has no transform {transform_name}"
        transforms_list.append(transform(**transform_args))
    return T.Compose(transforms_list)


class CelebA(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "celeba",
            image_shape: int = (128, 128),
            batch_size: int = 32,
            num_workers: int = 32,
            validation_len: tp.Union[int, float] = 500,
            train_transforms: tp.Optional[tp.Dict] = None,
            val_transforms: tp.Optional[tp.Dict] = None,
            **kwargs
    ):
        if train_transforms is not None:
            train_transforms = get_preprocessing_transforms(train_transforms)
        if val_transforms is not None:
            val_transforms = get_preprocessing_transforms(val_transforms)
        super(CelebA, self).__init__(train_transforms=train_transforms, val_transforms=val_transforms, **kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_len = validation_len
        self.image_shape = image_shape

    def setup(self, stage=None):
        self.celeba = torchvision.datasets.CelebA(self.data_dir, download=True)
        if isinstance(self.validation_len, float):
            self.validation_len = int(self.validation_len * len(self.celeba))
        self.train_dataset, self.val_dataset = random_split(self.celeba,
                                                            lengths=[len(self.celeba) - self.validation_len,
                                                                self.validation_len])
        self.train_dataset.dataset.transform = self.train_transforms
        self.val_dataset.dataset.transform = self.val_transforms

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
