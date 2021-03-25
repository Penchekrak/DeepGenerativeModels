import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything, Trainer
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path='configs')
def main(cfg: DictConfig):
    logger = WandbLogger(**cfg.logger)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    checkpoint = ModelCheckpoint(**cfg.checkpoint, dirpath=logger.save_dir)
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=checkpoint,
        plugins=DDPPlugin(find_unused_parameters=True)
    )
    model = instantiate(cfg.model, optimizer_conf=cfg.optimizer)
    datamodule = instantiate(cfg.datamodule)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    seed_everything(42)
    main()
