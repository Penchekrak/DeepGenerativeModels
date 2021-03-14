import torch
import wandb
from pytorch_lightning.metrics import Accuracy
from torch import nn
from utils import (permute_labels, calculate_fid, calculate_multilabel_accuracy,
                   create_generator_inputs, FidScore)
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid
import typing as tp
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=40, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=40, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / 2 ** repeat_num)
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class VanillaStarGAN(pl.LightningModule):

    def build_discriminator(self, image_shape: tp.Tuple[int]) -> nn.Module:
        return Discriminator()

    def build_generator(self, image_shape: tp.Tuple[int]) -> nn.Module:
        return Generator()

    def adversarial_loss(self, on_real_outputs: torch.Tensor,
                         on_fake_outputs: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = - torch.mean(on_real_outputs)
        if on_fake_outputs is not None:
            loss += torch.mean(on_fake_outputs)
        return loss

    def classification_loss(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(predictions, labels)

    def reconstruction_loss(self, reconstructed_images: torch.Tensor, original_images: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(original_images - reconstructed_images))

    def gradient_norm(self, outputs, inputs):
        weights = torch.ones_like(outputs, requires_grad=False)
        gradient = torch.autograd.grad(outputs=outputs,
                                       inputs=inputs,
                                       grad_outputs=weights,
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]
        gradient = torch.flatten(gradient, start_dim=1)
        return torch.sqrt(torch.sum(gradient ** 2, dim=1))

    def gradient_penalty(self, original_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand((original_images.shape[0], 1, 1, 1)).type_as(original_images)
        sampled_inputs = alpha * original_images.clone().detach().requires_grad_(True) + (
                1 - alpha) * fake_images.clone().detach().requires_grad_(True)
        discriminator_on_sampled_outputs, _ = self.discriminator(sampled_inputs)
        gradient_norm = self.gradient_norm(discriminator_on_sampled_outputs, sampled_inputs)
        return torch.mean((gradient_norm - 1) ** 2)

    # @abstractmethod
    def generator_loss(self, batch: tp.Tuple[torch.Tensor, torch.Tensor]) -> tp.Dict[str, tp.Any]:
        inputs, labels = batch
        labels = labels.float()
        permuted_labels = permute_labels(labels)

        generator_outputs = self.generator(inputs, permuted_labels)
        discriminator_on_fake_outputs, classification_on_fake_outputs = self.discriminator(generator_outputs)

        adversarial_loss = self.adversarial_loss(on_real_outputs=discriminator_on_fake_outputs)
        classification_loss = self.classification_loss(classification_on_fake_outputs, permuted_labels)

        reconstructed_inputs = self.generator(generator_outputs, labels)
        reconstruction_loss = self.reconstruction_loss(reconstructed_inputs, inputs)

        loss = adversarial_loss + self.lambda_reconstruction * reconstruction_loss + self.lambda_classification * classification_loss

        self.log_dict({
            'generator adversarial loss': adversarial_loss,
            'generator classification loss': classification_loss,
            'generator reconstruction loss': reconstruction_loss,
            'generator loss': loss
        })

        return loss

    # @abstractmethod
    def discriminator_loss(self, batch: tp.Tuple[torch.Tensor, torch.Tensor]) -> tp.Dict[str, tp.Any]:
        inputs, labels = batch
        labels = labels.float()
        permuted_labels = permute_labels(labels)

        discriminator_on_real_outputs, classification_on_real_outputs = self.discriminator(inputs)
        classification_loss = self.classification_loss(classification_on_real_outputs, labels)

        with torch.no_grad():
            generator_outputs = self.generator(inputs, permuted_labels)
        discriminator_on_fake_outputs, classification_on_fake_outputs = self.discriminator(generator_outputs)

        adversarial_loss = self.adversarial_loss(discriminator_on_real_outputs, discriminator_on_fake_outputs)
        gradient_penalty = self.gradient_penalty(inputs, generator_outputs)

        loss = adversarial_loss + self.lambda_gradient_penalty * gradient_penalty + self.lambda_classification * classification_loss

        self.log_dict({
            'discriminator adversarial loss': adversarial_loss,
            'discriminator classification loss': classification_loss,
            'discriminator gradient penalty': gradient_penalty,
            'discriminator loss': loss
        })
        return loss

    def __init__(
            self,
            control_images: torch.Tensor,
            # original_labels: tp.List[int],
            desired_labels: torch.Tensor,
            label_names: tp.Dict[int, str],
            discriminator_frequency: int = 5,
            lambda_reconstruction: float = 10.0,
            lambda_classification: float = 1.0,
            lambda_gradient_penalty: float = 10.0,
            *args, **kwargs
    ):
        super(VanillaStarGAN, self).__init__(*args, **kwargs)
        self.discriminator_frequency = discriminator_frequency
        self.control_images = control_images
        # self.original_labels = original_labels
        self.desired_labels = desired_labels
        self.label_names = label_names
        self.image_shape = control_images.shape[1:]
        self.discriminator = self.build_discriminator(self.image_shape)
        self.generator = self.build_generator(self.image_shape)
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_classification = lambda_classification
        self.lambda_gradient_penalty = lambda_gradient_penalty

        self.accuracy = Accuracy(compute_on_step=False)
        self.fid = FidScore(compute_on_step=False)

        # self.example_input_array = self.control_images

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        permuted_labels = permute_labels(labels)
        # desired_labels = self.desired_labels.type_as(images)
        # for label in desired_labels
        generator_outputs = self.generator(images, permuted_labels)
        discriminator_on_real_outputs, classification_on_real_outputs = self.discriminator(images)
        self.fid(images, generator_outputs)
        self.accuracy(labels, classification_on_real_outputs)
        # discriminator_on_fake_outputs, classification_on_fake_outputs = self.discriminator(generator_outputs)
        return {
            'real images': images if batch_idx == 0 else None,
        }

    def validation_epoch_end(self, outputs: tp.List[tp.Any]) -> None:
        n_images = 5
        for output in outputs:
            if output['real images'] is not None:
                control_images = output['real images'][0:n_images]
                break
        # fid_score = calculate_fid(self.generator, self.datamodule.val_dataloader)
        # discriminator_accuracy = calculate_multilabel_accuracy(self.discriminator, self.datamodule.val_dataloader)
        control_images = self.generate_images(control_images, self.desired_labels)  # , self.original_labels
        self.logger.experiment.log(
            {
                'fid_score': self.fid.compute(),
                'accuracy': self.accuracy.compute(),
                'control_images': control_images
            },
            step=self.current_epoch,
        )

    def training_step(
            self,
            batch: tp.Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            optimizer_idx: int
    ) -> tp.Dict[str, tp.Any]:
        if optimizer_idx == 0:
            return self.generator_loss(batch)
        if optimizer_idx == 1:
            return self.discriminator_loss(batch)

    @torch.no_grad()
    def generate_images(
            self,
            original_images: torch.Tensor,
            desired_labels: torch.Tensor
    ) -> wandb.Image:  # original_labels,
        #         device = next(iter(self.generator.parameters())).device
        original_images = original_images  # .to(device)
        batch_size, _, image_height, image_width = original_images.shape
        desired_labels = desired_labels.type_as(original_images)
        generated_images = []
        y_ticks_positions = [image_height // 2]
        y_ticks_labels = ['original']
        for label in desired_labels:
            generated_images.append(
                self.generator(
                    original_images, label.unsqueeze(0).repeat((batch_size, 1))
                )
            )
            y_ticks_positions.append(y_ticks_positions[-1] + image_height)
            y_ticks_labels.append('\n'.join(self.label_names[i] for i, l in enumerate(label) if l > 0))

        image_grid = make_grid(torch.cat(generated_images), nrow=batch_size, normalize=True, scale_each=True)
        plt.imshow(image_grid.permute(1, 2, 0).cpu().numpy())
        plt.yticks(y_ticks_positions, y_ticks_labels)
        plt.tick_params(
            axis='y',
            which='both',
            left=False
        )
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            labelbottom=False
        )
        return wandb.Image(plt)

    def configure_optimizers(self):
        lr = 1e-4

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return {'optimizer': opt_g, 'frequency': 1}, \
            {'optimizer': opt_d, 'frequency': self.discriminator_frequency}

    def forward(self, inputs):
        return self.generator(inputs)

# class VanillaStarGAN(StarGAN):
#
#     def build_discriminator(self, image_shape: tp.Tuple[int]) -> nn.Module:
#         pass
#
#     def build_generator(self, image_shape: tp.Tuple[int]) -> nn.Module:
#         pass
#
#     def generator_loss(self, batch: tp.Tuple[torch.Tensor, torch.Tensor]) -> tp.Dict[str, tp.Any]:
#         pass
#
#     def discriminator_loss(self, batch: tp.Tuple[torch.Tensor, torch.Tensor]) -> tp.Dict[str, tp.Any]:
#         pass
# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # YOUR CODE
#
#     def forward(self, x, labels):
#         # YOUR CODE
#
#
# class Critic(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # YOUR CODE
#
#     def forward(self, x):
#         # YOUR CODE
#
#
# class StarGAN:
#     def __init__(self):
#         self.G = Generator()
#         self.D = Critic()
#
#         # YOUR CODE
#
#     def train(self):
#         self.G.train()
#         self.D.train()
#
#     def eval(self):
#         self.G.eval()
#         self.D.eval()
#
#     def to(self, device):
#         self.D.to(device)
#         self.G.to(device)
#
#     def trainG(self, image, label):
#         # YOUR CODE
#
#     def trainD(self, image, label):
#         # YOUR CODE
#
#     def generate(self, image, label):
#         # YOUR CODE
