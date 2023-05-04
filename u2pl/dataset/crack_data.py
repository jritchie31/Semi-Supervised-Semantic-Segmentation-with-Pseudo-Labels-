import copy
import math
import os
import os.path
import random

import logging
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import augmentation as psp_trsform
from .base import BaseDataset
from .sampler import DistributedGivenIterationSampler


class crackData(BaseDataset):
    def __init__(self, data_root, data_list, trs_form, seed, n_sup, split="val"):
        logger = logging.getLogger("global")
        super(crackData, self).__init__(data_list)
        self.data_root = data_root
        self.transform = trs_form
        random.seed(seed)
        if len(self.list_sample) >= n_sup and split == "train":
            list_sample_temp = random.sample(self.list_sample, n_sup)
        elif len(self.list_sample) < n_sup and split == "train":
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat

            list_sample_temp = random.sample(self.list_sample, n_sup)
        else:
            list_sample_temp = self.list_sample

        self.list_sample = list_sample_temp
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        logger.info("# num_samples: {}".format(self.num_sample))

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample[index][0])
        label_path = os.path.join(self.data_root, self.list_sample[index][1])
        image = self.img_loader(image_path, "L")
        label = self.img_loader(label_path, "L")
        image, label = self.transform(image, label)

        # Return image path along with image and label
        return image[0], label[0, 0].long(), image_path

    def __len__(self):
        return self.num_sample


def build_transform(cfg):
    trs_form = []
    mean, std = cfg["mean"], cfg["std"]
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type)
        )
    if cfg.get("cutout", False):
        n_holes, length = cfg["cutout"]["n_holes"], cfg["cutout"]["length"]
        trs_form.append(psp_trsform.Cutout(n_holes=n_holes, length=length))
    if cfg.get("cutmix", False):
        n_holes, prop_range = cfg["cutmix"]["n_holes"], cfg["cutmix"]["prop_range"]
        trs_form.append(psp_trsform.Cutmix(prop_range=prop_range, n_holes=n_holes))

    return psp_trsform.Compose(trs_form)


def build_crackloader(split, all_cfg, seed=0, distributed=False):
    cfg_dset = all_cfg["dataset"]
    cfg_trainer = all_cfg["trainer"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 2975)

    # build transform
    trs_form = build_transform(cfg)
    dset = crackData(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split)

    # build sampler
    if distributed:
        sampler = DistributedSampler(dset)
    else:
        sampler = torch.utils.data.RandomSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sampler,
        shuffle=False,
        pin_memory=False,
    )
    return loader


def build_crack_semi_loader(split, all_cfg, seed=0, distributed=False):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 1132)
    # build transform
    trs_form = build_transform(cfg)
    trs_form_unsup = build_transform(cfg)
    dset = crackData(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split)

    if split == "val":
        # build sampler
        if distributed:
            sampler = DistributedSampler(dset)
        else:
            sampler = torch.utils.data.RandomSampler(dset)

        loader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        # build sampler for unlabeled set
        data_list_unsup = cfg["data_list"].replace("labeled.txt", "unlabeled.txt")
        dset_unsup = crackData(
            cfg["data_root"], data_list_unsup, trs_form_unsup, seed, n_sup, split
        )

        if distributed:
            sampler_sup = DistributedSampler(dset)
            sampler_unsup = DistributedSampler(dset_unsup)
        else:
            sampler_sup = torch.utils.data.RandomSampler(dset)
            sampler_unsup = torch.utils.data.RandomSampler(dset_unsup)

        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler_sup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup
    
def build_crack_semi_portion_loader(split, all_cfg, seed=0, distributed=False):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    cfg_dlst = cfg['train']["data_list"]
    cfg_dlst_val = cfg['val']["data_list"]
    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sample = 1132
    n_val = 80
    p_sup = cfg.get("p_sup", 0.5)
    p_unsup = 1 - p_sup
    n_sup = round(n_sample * p_sup)
    n_unsup = n_sample - n_sup
    labeled_list, unlabeled_list = split_list(cfg_dlst, p_sup)
    # build transform
    trs_form = build_transform(cfg)
    trs_form_unsup = build_transform(cfg)
    
    dset_val = crackData(cfg['val']["data_root"], cfg_dlst_val, trs_form, seed, n_val, split)
    if cfg_dset["p_sup"] != 0:
        dset = crackData(cfg['train']["data_root"], labeled_list, trs_form, seed, n_sup, split)

    if split == "val":
        # build sampler
        if distributed:
            sampler = DistributedSampler(dset_val)
        else:
            sampler = torch.utils.data.RandomSampler(dset_val)

        loader = DataLoader(
            dset_val,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        if cfg_dset["p_sup"] == 0:
            # build sampler for unlabeled set
            dset_unsup = crackData(
                cfg["data_root"], unlabeled_list, trs_form_unsup, seed, n_unsup, split
            )

            if distributed:
                sampler_unsup = DistributedSampler(dset_unsup)
            else:
                sampler_unsup = torch.utils.data.RandomSampler(dset_unsup)

            loader_unsup = DataLoader(
                dset_unsup,
                batch_size=batch_size,
                num_workers=workers,
                sampler=sampler_unsup,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
            )
            return loader_unsup           
        elif cfg_dset["p_sup"] == 1:
            if distributed:
                sampler_sup = DistributedSampler(dset)
            else:
                sampler_sup = torch.utils.data.RandomSampler(dset)

            loader_sup = DataLoader(
                dset,
                batch_size=batch_size,
                num_workers=workers,
                sampler=sampler_sup,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
            )
            return loader_sup
        else:
            # build sampler for unlabeled set
            dset_unsup = crackData(
                cfg["data_root"], unlabeled_list, trs_form_unsup, seed, n_unsup, split
            )

            if distributed:
                sampler_sup = DistributedSampler(dset)
                sampler_unsup = DistributedSampler(dset_unsup)
            else:
                sampler_sup = torch.utils.data.RandomSampler(dset)
                sampler_unsup = torch.utils.data.RandomSampler(dset_unsup)

            loader_sup = DataLoader(
                dset,
                batch_size=batch_size,
                num_workers=workers,
                sampler=sampler_sup,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
            )

            loader_unsup = DataLoader(
                dset_unsup,
                batch_size=batch_size,
                num_workers=workers,
                sampler=sampler_unsup,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
            )
            return loader_sup, loader_unsup

def build_crack_semi_active_loader(split, all_cfg, seed=0, distributed=False):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    cfg_all_dlst = cfg["train"]["data_list"]
    cfg_sup_dlst = cfg["train"]["ann_data_list"]
    save_dir = os.path.dirname(cfg_sup_dlst)
    
    cfg_uns_dlst, n_sup, n_unsup = create_unsupervised_data_list(cfg_all_dlst, cfg_sup_dlst, save_dir)
    cfg_dlst_val = cfg['val']["data_list"]
    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_val = 80
    # build transform
    trs_form = build_transform(cfg)
    trs_form_unsup = build_transform(cfg)
    
    dset_val = crackData(cfg['val']["data_root"], cfg_dlst_val, trs_form, seed, n_val, split)
    dset = crackData(cfg['train']["data_root"], cfg_sup_dlst, trs_form, seed, n_sup, split)

    if split == "val":
        # build sampler
        if distributed:
            sampler = DistributedSampler(dset_val)
        else:
            sampler = torch.utils.data.RandomSampler(dset_val)

        loader = DataLoader(
            dset_val,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        # build sampler for unlabeled set
        dset_unsup = crackData(
            cfg["data_root"], cfg_uns_dlst, trs_form_unsup, seed, n_unsup, split
        )

        if distributed:
            sampler_sup = DistributedSampler(dset)
            sampler_unsup = DistributedSampler(dset_unsup)
        else:
            sampler_sup = torch.utils.data.RandomSampler(dset)
            sampler_unsup = torch.utils.data.RandomSampler(dset_unsup)

        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler_sup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup

def split_list(cfg_dlst, p_sup):
    # Read the lines from the original list file
    with open(cfg_dlst, 'r') as f:
        lines = f.readlines()

    p_unsup = 1 - p_sup
    # Shuffle the lines
    random.shuffle(lines)
    n_sample = len(lines)
    n_sup = int(n_sample * p_sup)

    # Split the lines into two separate lists
    labeled_lines = lines[:n_sup]
    unlabeled_lines = lines[n_sup:]

    # Save the new lists to the same directory as the original file
    dir_path = os.path.dirname(cfg_dlst)
    labeled_list = os.path.join(dir_path, f'labeled_portion({p_sup}-{p_unsup}).txt')
    unlabeled_list = os.path.join(dir_path, f'unlabeled_portion({p_sup}-{p_unsup}).txt')

    with open(labeled_list, 'w') as f:
        f.writelines(labeled_lines)

    with open(unlabeled_list, 'w') as f:
        f.writelines(unlabeled_lines)

    return labeled_list, unlabeled_list

def create_unsupervised_data_list(cfg_all_dlst, cfg_sup_dlst, save_dir):
    with open(cfg_all_dlst, 'r') as f:
        all_data = set(line.strip() for line in f)

    with open(cfg_sup_dlst, 'r') as f:
        sup_data = set(line.strip() for line in f)

    unsup_data = all_data - sup_data
    cfg_uns_dlst = os.path.join(save_dir, "cfg_uns_dlst.txt")

    with open(cfg_uns_dlst, 'w') as f:
        for path in unsup_data:
            f.write(f"{path}\n")

    n_sup = len(sup_data)
    n_unsup = len(unsup_data)
    n_all = len(all_data)

    assert n_sup + n_unsup == n_all, f"Unexpected number of samples, found {n_sup + n_unsup}, expected {n_all}"

    return cfg_uns_dlst, n_sup, n_unsup