import logging

from .cityscapes import build_city_semi_loader, build_cityloader
from .crack_data import build_crackloader, build_crack_semi_loader
from .pascal_voc import build_voc_semi_loader, build_vocloader

logger = logging.getLogger("global")
#Function checks the type of dataset specified in the configuration, and based on type will call one of the
#loader functions to build and return appropriate data loader for training/val
#A data loader is an iteravle that loads dataset in batches making it available for training or testing a ML model


def get_loader(cfg, seed=0, distributed=False):
    cfg_dataset = cfg["dataset"]

    if cfg_dataset["type"] == "cityscapes_semi":
        train_loader_sup, train_loader_unsup = build_city_semi_loader(
            "train", cfg, seed=seed, distributed=distributed
        )
        val_loader = build_cityloader("val", cfg, distributed=distributed)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader

    elif cfg_dataset["type"] == "cityscapes":
        train_loader_sup = build_cityloader("train", cfg, seed=seed, distributed=distributed)
        val_loader = build_cityloader("val", cfg, distributed=distributed)
        logger.info("Get loader Done...")
        return train_loader_sup, val_loader

    elif cfg_dataset["type"] == "pascal_semi":
        train_loader_sup, train_loader_unsup = build_voc_semi_loader(
            "train", cfg, seed=seed, distributed=distributed
        )
        val_loader = build_vocloader("val", cfg, distributed=distributed)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader

    elif cfg_dataset["type"] == "pascal":
        train_loader_sup = build_vocloader("train", cfg, seed=seed, distributed=distributed)
        val_loader = build_vocloader("val", cfg, distributed=distributed)
        logger.info("Get loader Done...")
        return train_loader_sup, val_loader

    elif cfg_dataset["type"] == "crack_semi":
        train_loader_sup, train_loader_unsup = build_crack_semi_loader(
            "train", cfg, seed=seed, distributed=distributed
        )
        val_loader = build_vocloader("val", cfg, distributed=distributed)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader

    elif cfg_dataset["type"] == "crack":
        train_loader_sup = build_crackloader("train", cfg, seed=seed, distributed=distributed)
        val_loader = build_vocloader("val", cfg, distributed=distributed)
        logger.info("Get loader Done...")
        return train_loader_sup, val_loader

    else:
        raise NotImplementedError(
            "dataset type {} is not supported".format(cfg_dataset)
        )
