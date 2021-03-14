import torch
from torch import nn
from utils import compute_gradient_penalty, optimizer_to
from block import Block, ResBlock


class Generator(nn.Module):
    def __init__(self, domain_dim: int, begin_conv_dim: int):
        super().__init__()
        self.begin_layers = Block(domain_dim + 3, begin_conv_dim,
                                  kernel_size=7, stride=1,
                                  padding=3, in_norm=True)
        self.downsampling = nn.Sequential(*[Block(begin_conv_dim * 2 ** i,
                                                  begin_conv_dim * 2 ** (i + 1),
                                                  kernel_size=4, stride=2,
                                                  padding=1, in_norm=True)
                                            for i in range(2)])

        self.bottleneck = nn.Sequential(*[ResBlock(begin_conv_dim * 4, begin_conv_dim * 4)
                                          for _ in range(6)])
        self.upsampling = nn.Sequential(*[Block(begin_conv_dim * 4 // 2 ** i,
                                                begin_conv_dim * 4 // 2 ** (i + 1),
                                                kernel_size=4, stride=2, padding=1,
                                                in_norm=self.begin_layers, conv_tr=True)
                                          for i in range(2)])
        self.end_layers = nn.Sequential(nn.Conv2d(begin_conv_dim, 3,
                                                  kernel_size=7, stride=1, padding=3),
                                        nn.Tanh())

    def forward(self, x: torch.tensor, labels: torch.tensor):
        labels = labels.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, labels.repeat(1, 1, *x.shape[2:])], dim=1)
        x = self.begin_layers(x)
        x = self.downsampling(x)
        x = self.bottleneck(x)
        x = self.upsampling(x)
        x = self.end_layers(x)
        return x


class Critic(nn.Module):
    def __init__(self, domain_dim: int, begin_conv_dim: int,
                 image_dim: int):
        super().__init__()
        self.begin_layers = Block(3, begin_conv_dim,
                                  kernel_size=4, stride=2,
                                  padding=1, leaky=True)
        self.hidden = nn.Sequential(*[Block(begin_conv_dim * 2 ** i,
                                            begin_conv_dim * 2 ** (i + 1),
                                            kernel_size=4, stride=2,
                                            padding=1, leaky=True)
                                      for i in range(5)])
        self.end_layers1 = nn.Conv2d(begin_conv_dim * 2 ** 5, 1,
                                     kernel_size=3, stride=1, padding=1)
        self.end_layers2 = nn.Conv2d(begin_conv_dim * 2 ** 5, domain_dim,
                                     kernel_size=image_dim // 64,
                                     stride=1, padding=0)

    def forward(self, x: torch.tensor):
        x = self.begin_layers(x)
        x = self.hidden(x)
        # src and cls
        res_src = self.end_layers1(x)
        res_cls = self.end_layers2(x)
        return res_src, res_cls.view(res_cls.shape[:2])


class StarGAN:
    def __init__(self,
                 domain_dim: int,
                 begin_conv_dim: int,
                 image_dim: int,
                 G_lr: float = 1e-4,
                 D_lr: float = 1e-4,
                 lambda_cls: float = 1.,
                 lambda_rec: float = 10.,
                 lambda_gp: float = 10.):
        self.lambda_rec = lambda_rec
        self.lambda_cls = lambda_cls
        self.lambda_gp = lambda_gp
        self.G = Generator(domain_dim, begin_conv_dim)
        self.D = Critic(domain_dim, begin_conv_dim, image_dim)
        # self.G = Generator()
        # self.D = Critic()
        self.G_opt = torch.optim.Adam(self.G.parameters(),
                                      lr=G_lr, betas=(0.5, 0.999))
        self.D_opt = torch.optim.Adam(self.D.parameters(),
                                      lr=D_lr, betas=(0.5, 0.999))
        self.cls_criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.rec_criterion = nn.L1Loss()

        self.lam_cls, self.lam_rec, self.lam_gp = 1, 10, 10

    def save(self, path: str):
        torch.save({'D_state_dict': self.D.state_dict(),
                    'G_state_dict': self.G.state_dict(),
                    'D_opt_state_dict': self.D_opt.state_dict(),
                    'G_opt_state_dict': self.G_opt.state_dict()
                    }, path)

    def load(self, path: str):
        device = next(iter(self.G.parameters())).device()
        checkpoint = torch.load(path)
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.G_opt.load_state_dict(checkpoint['G_opt_state_dict'])
        self.D_opt.load_state_dict(checkpoint['D_opt_state_dict'])

    def train(self):
        self.G.train()
        self.D.train()

    def eval(self):
        self.G.eval()
        self.D.eval()

    def to(self, device):
        self.D.to(device)
        self.G.to(device)
        optimizer_to(self.D_opt, device)
        optimizer_to(self.G_opt, device)

    def get_device(self):
        return next(self.D.parameters()).device

    def trainG(self, real_image: torch.Tensor,
               real_label: torch.Tensor, fake_label: torch.Tensor):
        fake_image = self.G(real_image, fake_label)
        fake_score, fake_pred_label = self.D(fake_image)
        adv_loss = -torch.mean(fake_score)

        cls_loss = self.cls_criterion(fake_pred_label,
                                      fake_label.float()
                                      ) / fake_label.shape[0]

        recover_image = self.G(fake_image, real_label)
        rec_loss = self.rec_criterion(recover_image, real_image)

        full_loss = (adv_loss +
                     self.lambda_cls * cls_loss +
                     self.lambda_rec * rec_loss
                     )
        self.G_opt.zero_grad()
        full_loss.backward()
        self.G_opt.step()
        return {'G_full_loss': full_loss.item(),
                'G_adv_loss': adv_loss.item(),
                'G_cls_loss': (self.lambda_cls * cls_loss).item(),
                'G_rec_loss': (self.lambda_rec * rec_loss).item()
                }

    def trainD(self, real_image: torch.tensor,
               real_label: torch.tensor, fake_label: torch.tensor):
        with torch.no_grad():
            fake_image = self.G(real_image, fake_label)
        real_score, real_pred_label = self.D(real_image)
        fake_score, fake_pred_label = self.D(fake_image)

        cls_loss = self.cls_criterion(real_pred_label,
                                      real_label.float()
                                      ) / real_label.shape[0]

        gradient_penalty = compute_gradient_penalty(self.D, real_image,
                                                    fake_image)
        adv_loss = (- torch.mean(real_score)
                    + torch.mean(fake_score)
                    + self.lambda_gp * gradient_penalty)

        full_loss = adv_loss + self.lambda_cls * cls_loss
        self.D_opt.zero_grad()
        full_loss.backward()
        self.D_opt.step()
        return {'D_full_loss': full_loss.item(),
                'D_adv_loss': adv_loss.item(),
                'D_cls_loss': (self.lambda_cls * cls_loss).item()
                }
