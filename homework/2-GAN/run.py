import os
import wandb
from tqdm.notebook import tqdm, trange
from model import StarGAN
from utils import DataPrep, permute_labels
from calculate_fid import CalcFID


def train(stargan: StarGAN, data: DataPrep, n_epoch: int, n_critic: int, name: str):
    run = wandb.init(name=name,
                     project='gen_models_hw2')
    path = 'parameters/' + name
    if not os.path.exists(path):
        os.makedirs(path)
    last_itr = 0
    fid_calculator = CalcFID()
    for epoch in trange(1, desc='train loop', leave=True):
        device = stargan.get_device()
        for itr, (image, attr) in enumerate(
                tqdm(data.loader, desc=f'epoch {epoch}', leave=False)):
            last_itr += 1
            image = image.to(device)
            label = data.get_labels(attr).to(device)
            fake_label = permute_labels(label)
            # Train D
            D_losses = stargan.trainD(image, label.float(), fake_label)
            # print(D_losses)
            if not (itr + 1) % n_critic:
                G_losses = stargan.trainG(image, label.float(), fake_label.float())
                wandb.log({'itr': last_itr})
                wandb.log(D_losses)
                wandb.log(G_losses)
        fid = fid_calculator(data.loader, stargan)
        wandb.log({'epoch': epoch + 1, 'FID': fid})
