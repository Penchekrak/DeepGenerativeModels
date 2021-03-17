import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from model import VanillaStarGAN
from datamodules import CelebaDataModule


def main(args):
    logger = WandbLogger(project=args.project_name, save_dir=None, log_model=True)
    model_checkpointer = ModelCheckpoint(dirpath=logger.save_dir, monitor=args.monitor)
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=[model_checkpointer],
                                         plugins=DDPPlugin(find_unused_parameters=True))
    celeba = CelebaDataModule.from_argparse_args(args)
    model = VanillaStarGAN.load_from_checkpoint(args.checkpoint, image_shape=celeba.image_shape,
                                                label_names=celeba.attributes)
    trainer.fit(model, datamodule=celeba)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--project_name', default='GAN-homework_2-GAN')
    parser.add_argument('--monitor', default='fid score')
    parser.add_argument('--checkpoint')
    parser = VanillaStarGAN.add_argparse_args(parser)
    parser = CelebaDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    main(args)
