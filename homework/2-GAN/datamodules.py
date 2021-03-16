import torchvision
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule
import typing as tp


class CelebaDataModule(LightningDataModule):

    def __init__(
            self,
            data_dir: str = "celeba",
            image_shape: int = (128, 128),
            batch_size: int = 32,
            num_workers: int = 32,
            validation_len: tp.Union[int, float] = 500,
            attributes: tp.List[str] = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Wearing_Hat',
                'Mustache'],
            **kwargs
    ):
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        attrs = '5_o_Clock_Shadow ' \
                'Arched_Eyebrows ' \
                'Attractive ' \
                'Bags_Under_Eyes ' \
                'Bald ' \
                'Bangs ' \
                'Big_Lips ' \
                'Big_Nose ' \
                'Black_Hair ' \
                'Blond_Hair ' \
                'Blurry ' \
                'Brown_Hair ' \
                'Bushy_Eyebrows ' \
                'Chubby ' \
                'Double_Chin ' \
                'Eyeglasses ' \
                'Goatee ' \
                'Gray_Hair ' \
                'Heavy_Makeup ' \
                'High_Cheekbones ' \
                'Male ' \
                'Mouth_Slightly_Open ' \
                'Mustache ' \
                'Narrow_Eyes ' \
                'No_Beard ' \
                'Oval_Face ' \
                'Pale_Skin ' \
                'Pointy_Nose ' \
                'Receding_Hairline ' \
                'Rosy_Cheeks ' \
                'Sideburns ' \
                'Smiling ' \
                'Straight_Hair ' \
                'Wavy_Hair ' \
                'Wearing_Earrings ' \
                'Wearing_Hat ' \
                'Wearing_Lipstick ' \
                'Wearing_Necklace ' \
                'Wearing_Necktie ' \
                'Young'
        attr2index = {j: i for i, j in enumerate(attrs.split())}
        self.attributes = attributes
        self.label_subset_indices = [attr2index[j] for j in self.attributes]
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_len = validation_len
        self.image_shape = image_shape

    def target_transform(self, label):
        return label[self.label_subset_indices]

    def setup(self, stage=None):
        self.celeba = torchvision.datasets.CelebA(self.data_dir, target_type='attr', transform=self.transforms,
                                                  target_transform=self.target_transform, download=True)
        if isinstance(self.validation_len, float):
            self.validation_len = int(self.validation_len * len(self.celeba))
        self.train_dataset, self.val_dataset = random_split(self.celeba,
                                                            lengths=[len(self.celeba) - self.validation_len,
                                                                self.validation_len])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
