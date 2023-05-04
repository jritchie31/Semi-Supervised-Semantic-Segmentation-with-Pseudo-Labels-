import argparse
import copy
import logging
import os
import os.path as osp
import pprint
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from itertools import cycle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter

from u2pl.dataset.augmentation import generate_unsup_data
from u2pl.dataset.builder import get_loader
from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.dist_helper import setup_distributed
from u2pl.utils.loss_helper import (
    compute_contra_memobank_loss,
    compute_unsupervised_loss,
    get_criterion,
)
from u2pl.utils.lr_helper import get_optimizer, get_scheduler
from u2pl.utils.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    label_onehot,
    load_state,
    set_random_seed,
)

# Get the absolute path of the current file
current_dir = osp.dirname(osp.abspath(__file__))
experiment_config_dir = osp.join(current_dir, r"experiments/data_crack/ours/config_local.yaml") #Need to change this to crack

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default=experiment_config_dir)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--port", default=None, type=int)
# Add a new command line argument for distributed training
parser.add_argument("--distributed", action="store_true", help="Enable distributed training")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # macOS
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def to_device(tensor):
    return tensor.to(device)

def main():
    global args, cfg, prototype
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    seed = cfg["dataset"]["train"]["seed"]
    cfg["exp_path"] = os.path.dirname(args.config)
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    cudnn.enabled = True
    cudnn.benchmark = True

    # Check if distributed training is enabled
    if args.distributed:
        rank, world_size = setup_distributed(port=args.port)
    else:
        rank, world_size = 0, 1

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_logger = SummaryWriter(
            osp.join(cfg["exp_path"], "log/events_seg/" + current_time)
        )
    else:
        tb_logger = None

    if seed is not None:
        print("set random seed to", seed)
        set_random_seed(seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]) and rank == 0:
        os.makedirs(cfg["saver"]["snapshot_dir"])
    
    if cfg["dataset"]["p_sup"] == 0:
        cfg["dataset"]["batch_size"] = 1
        train_loader_unsup, val_loader = get_loader(cfg, seed=seed, distributed=args.distributed)
        
        # Optimizer and lr decay scheduler
        cfg_trainer = cfg["trainer"]
        cfg_optim = cfg_trainer["optimizer"]
        times = 1

        # Create network
        model = ModelBuilder(cfg["net"])
        modules_back = [model.encoder]
        if cfg["net"].get("aux_loss", False):
            modules_head = [model.auxor, model.decoder]
        else:
            modules_head = [model.decoder]

        if cfg["net"].get("sync_bn", True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = to_device(model)
        # Teacher model
        model_teacher = ModelBuilder(cfg["net"])
        model_teacher = to_device(model_teacher)
        if args.distributed:
            model_teacher = torch.nn.parallel.DistributedDataParallel(
                model_teacher,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        for p in model_teacher.parameters():
            p.requires_grad = False    

        modules_back = [model_teacher.encoder]
        if cfg["net"].get("aux_loss", False):
            modules_head = [model_teacher.auxor, model_teacher.decoder]
        else:
            modules_head = [model_teacher.decoder]

        params_list = []
        for module in modules_back:
            params_list.append(
                dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
            )
        for module in modules_head:
            params_list.append(
                dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
            )
    
    elif cfg["dataset"]["p_sup"] == 1:
        # Create network
        model = ModelBuilder(cfg["net"])
        modules_back = [model.encoder]
        if cfg["net"].get("aux_loss", False):
            modules_head = [model.auxor, model.decoder]
        else:
            modules_head = [model.decoder]

        if cfg["net"].get("sync_bn", True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = to_device(model)

        sup_loss_fn = get_criterion(cfg)

        train_loader_sup, val_loader = get_loader(cfg, seed=seed, distributed=args.distributed)

        # Optimizer and lr decay scheduler
        cfg_trainer = cfg["trainer"]
        cfg_optim = cfg_trainer["optimizer"]
        times = 1

        params_list = []
        for module in modules_back:
            params_list.append(
                dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
            )
        for module in modules_head:
            params_list.append(
                dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
            )

        optimizer = get_optimizer(params_list, cfg_optim)

        if args.distributed:
            local_rank = int(os.environ["LOCAL_RANK"])
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
    
    else:
        # Create network
        model = ModelBuilder(cfg["net"])
        modules_back = [model.encoder]
        if cfg["net"].get("aux_loss", False):
            modules_head = [model.auxor, model.decoder]
        else:
            modules_head = [model.decoder]

        if cfg["net"].get("sync_bn", True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = to_device(model)

        sup_loss_fn = get_criterion(cfg)

        train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed, distributed=args.distributed)

        # Optimizer and lr decay scheduler
        cfg_trainer = cfg["trainer"]
        cfg_optim = cfg_trainer["optimizer"]
        times = 1

        params_list = []
        for module in modules_back:
            params_list.append(
                dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
            )
        for module in modules_head:
            params_list.append(
                dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
            )

        optimizer = get_optimizer(params_list, cfg_optim)

        if args.distributed:
            local_rank = int(os.environ["LOCAL_RANK"])
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        # Teacher model
        model_teacher = ModelBuilder(cfg["net"])
        model_teacher = to_device(model_teacher)
        if args.distributed:
            model_teacher = torch.nn.parallel.DistributedDataParallel(
                model_teacher,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        for p in model_teacher.parameters():
            p.requires_grad = False

    best_prec = 0
    last_epoch = 0

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(cfg["saver"]["pretrain"], model_teacher, key="teacher_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    if cfg["dataset"]["p_sup"] == 0:
        lr_scheduler = get_scheduler(
            cfg_trainer, len(train_loader_unsup), optimizer_start, start_epoch=last_epoch
                )       
    elif cfg["dataset"]["p_sup"] == 1:
        lr_scheduler = get_scheduler(
            cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
                ) 
    else:
        max_length = max(len(train_loader_sup), len(train_loader_unsup))
        lr_scheduler = get_scheduler(
            cfg_trainer, max_length, optimizer_start, start_epoch=last_epoch
                )

    # build prototype
    prototype = torch.zeros(
        (
            cfg["net"]["num_classes"],
            cfg["trainer"]["contrastive"]["num_queries"],
            1,
            256,
        )
    )
    prototype = to_device(prototype)

    # Call select_diverse_samples after a few epochs (e.g., 5)
    if cfg["dataset"]["p_sup"] == 0:
        # Set the number of clusters and samples per cluster
        # Call the select_diverse_samples function
        extract_features(model_teacher, train_loader_unsup, device, logger)
        logger.info("Generated the list of features and filenames, please process")
        # Save the selected paths to a file

    elif cfg["dataset"]["p_sup"] == 1:
        for epoch in range(last_epoch, cfg_trainer["epochs"]):
            # Training
            sup_train(
                model,
                optimizer,
                lr_scheduler,
                sup_loss_fn,
                train_loader_sup,
                epoch,
                tb_logger,
                logger,
            )

            # Validation and store checkpoint
            prec = validate(model, val_loader, epoch, logger, device)

            if rank == 0:
                state = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_miou": best_prec,
                }

                if prec > best_prec:
                    best_prec = prec
                    state["best_miou"] = prec
                    torch.save(
                        state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth")
                    )

                torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt.pth"))

                logger.info(
                    "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                        best_prec * 100
                    )
                )
                tb_logger.add_scalar("mIoU val", prec, epoch)
    else:
        # Start to train model
        for epoch in range(last_epoch, cfg_trainer["epochs"]):
            # Training
            train(
                model,
                model_teacher,
                optimizer,
                lr_scheduler,
                sup_loss_fn,
                train_loader_sup,
                train_loader_unsup,
                epoch,
                tb_logger,
                logger,
            )

            # Validation
            if cfg_trainer["eval_on"]:
                if rank == 0:
                    logger.info("start evaluation")

                if epoch < cfg["trainer"].get("sup_only_epoch", 1):
                    prec = validate(model, val_loader, epoch, logger, device)
                else:
                    prec = validate(model_teacher, val_loader, epoch, logger, device)

                if rank == 0:
                    state = {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "teacher_state": model_teacher.state_dict(),
                        "best_miou": best_prec,
                    }
                    if prec > best_prec:
                        best_prec = prec
                        torch.save(
                            state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth")
                        )

                    torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt.pth"))

                    logger.info(
                        "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                            best_prec * 100
                        )
                    )
                    tb_logger.add_scalar("mIoU val", prec, epoch)
        
        teacher_entropies, image_filenames = get_teacher_predictions(model_teacher, train_loader_unsup, device, logger)
        # Create a new list to store the filenames of the most annotation-needed images.
        annotation_needed_filenames = []
        with torch.no_grad():            
            # Convert the teacher_entropies list to a tensor
            entropy_tensor = torch.tensor(teacher_entropies)
            # Assuming that image_filenames is a list of image filenames corresponding to pred_u_large_teacher
            # Replace N with the number of top images you want to select
            N = len(image_filenames)
            # Get the top N entropy values and their indices
            top_entropy_values, top_entropy_indices = torch.topk(entropy_tensor, N, largest=True)
            
            # Return the list of filenames.
            for idx in top_entropy_indices:
                annotation_needed_filenames.append(image_filenames[idx])
        # Save the list of filenames as a .txt file in the checkpoints folder
        filenames_string = "\n".join(annotation_needed_filenames)

        ckpt_dir = osp.join(current_dir, cfg["saver"]["snapshot_dir"])
        result_dir = osp.join(ckpt_dir, f"results")
        # Create the result_dir folder if it doesn't exist
        os.makedirs(result_dir, exist_ok=True)
        pred_teacher_filepath = osp.join(result_dir, f"annotation_queue.txt")

        with open(pred_teacher_filepath, "w") as pred_teacher_file:
            pred_teacher_file.write(filenames_string)

def extract_features(model, dataloader, device, logger):
    model.eval()
    means = []
    stds = []
    paths = []

    with torch.no_grad():
        loader_u_iter = iter(dataloader)
        for step in range(len(loader_u_iter)):
            image_u, _, path_u = next(loader_u_iter)
            inputs = image_u.to(device)
            # Assuming that your model returns a dictionary with keys 'pred', 'rep', and 'aux'
            pred_teacher = model(inputs)['pred']
            prob = F.softmax(pred_teacher, dim=1)
            class_1_prob = prob[:, 1, :, :].cpu().numpy()

            mean = np.mean(class_1_prob, axis=(1, 2))
            std = np.std(class_1_prob, axis=(1, 2))
            mean = mean.astype(np.float64)
            mean = np.round(mean,5)
            std = std.astype(np.float64)
            std = np.round(std,5)

            means.append(mean)
            stds.append(std)
            paths.extend(path_u)

            if step % 100 == 0:
                logger.info(
                    "[Extracting Features]"
                    "Num [{}/{}]\t".format(
                        step,
                        len(loader_u_iter),
                    )
                )

    means = np.concatenate(means, axis=0)
    stds = np.concatenate(stds, axis=0)
    features = np.stack((means, stds), axis=-1)

    # Save features and paths to text files
    checkpoint_dir = cfg["saver"]["snapshot_dir"]
    np.savetxt(f"{checkpoint_dir}/features.txt", features, fmt="%.5f", delimiter=",")
    with open(f"{checkpoint_dir}/paths.txt", "w") as file:
        for path in paths:
            file.write(f"{path}\n")

    return features, paths

def sup_train(
    model,
    optimizer,
    lr_scheduler,
    criterion,
    data_loader,
    epoch,
    tb_logger,
    logger,
    distributed = False,
):
    model.train()
    if distributed:
        data_loader.sampler.set_epoch(epoch)
    data_loader_iter = iter(data_loader)

    if distributed:
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1
    
    losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    for step in range(len(data_loader)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(data_loader) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        image, label, _ = next(data_loader_iter)
        batch_size, h, w = label.size()
        image, label = to_device(image), to_device(label)
        outs = model(image)
        pred = outs["pred"]
        pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

        if "aux_loss" in cfg["net"].keys():
            aux = outs["aux"]
            aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
            loss = criterion([pred, aux], label)
        else:
            loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # gather all loss from different gpus
        reduced_loss = loss.clone().detach()
        if distributed:
            dist.all_reduce(reduced_loss)
        losses.update(reduced_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 10 == 0 and rank == 0:
            logger.info(
                "[Supervised]"
                "Iter [{}/{}]\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "LR {lr.val:.5f} ({lr.avg:.5f})\t".format(
                    i_iter,
                    cfg["trainer"]["epochs"] * len(data_loader),
                    data_time=data_times,
                    batch_time=batch_times,
                    loss=losses,
                    lr=learning_rates,
                )
            )

            tb_logger.add_scalar("lr", learning_rates.avg, i_iter)
            tb_logger.add_scalar("Loss", losses.avg, i_iter)

def train(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    distributed=False,
):
    global prototype
    ema_decay_origin = cfg["net"]["ema_decay"]

    #a = len(loader_l)
    #b = len(loader_u)
    max_length = max(len(loader_l), len(loader_u))
    model.train()

    if distributed:
        loader_l.sampler.set_epoch(epoch)
        loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    loader_l_circular = cycle(loader_l)
    loader_u_circular = cycle(loader_u)
    """assert len(loader_l) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"
    """
    if distributed:
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    for step in range(max_length):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * max_length + step  # Update i_iter to use max_length
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        # Use circular iterators instead of regular iterators
        image_l, label_l, _ = next(loader_l_circular)
        batch_size, h, w = label_l.size()

        # Show the images and labels
        #visualize_image_label_batch(image_l, label_l, batch_size)
        
        image_l, label_l = to_device(image_l), to_device(label_l)

        image_u, _, _ = next(loader_u_circular)
        # Show the images and labels
        #visualize_image_label_batch(image_u, _, batch_size)
        image_u = to_device(image_u)

        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            #contra_flag = "none"
            # forward
            outs = model(image_l)
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred, aux], label_l)
            else:
                sup_loss = sup_loss_fn(pred, label_l)

            model_teacher.train()
            _ = model_teacher(image_l)

            unsup_loss = 0 * rep.sum()
            contra_loss = 0 * rep.sum()
        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1):
                # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data

            # generate pseudo labels first
            model_teacher.eval()
            pred_u_teacher = model_teacher(image_u)["pred"]
            pred_u_teacher = F.interpolate(
                pred_u_teacher, (h, w), mode="bilinear", align_corners=True
            )
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
            logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

            # apply strong data augmentation: cutout, cutmix, or classmix
            if np.random.uniform(0, 1) < 0.5 and cfg["trainer"]["unsupervised"].get(
                "apply_aug", False
            ):
                image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                    image_u,
                    label_u_aug.clone(),
                    logits_u_aug.clone(),
                    mode=cfg["trainer"]["unsupervised"]["apply_aug"],
                )
            else:
                image_u_aug = image_u

            # forward
            num_labeled = len(image_l)
            image_all = torch.cat((image_l, image_u_aug))
            
            outs = model(image_all)
            pred_all, rep_all = outs["pred"], outs["rep"]
            pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]
            pred_l_large = F.interpolate(
                pred_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_u_large = F.interpolate(
                pred_u, size=(h, w), mode="bilinear", align_corners=True
            )

            # Call the function with pred_l_large and pred_u_large as input arguments
            #visualize_predictions(pred_l_large, pred_u_large)
            
            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"][:num_labeled]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred_l_large, aux], label_l.clone())
            else:
                sup_loss = sup_loss_fn(pred_l_large, label_l.clone())

            # teacher forward
            model_teacher.train()
            with torch.no_grad():
                out_t = model_teacher(image_all)
                pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
                prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
                prob_l_teacher, prob_u_teacher = (
                    prob_all_teacher[:num_labeled],
                    prob_all_teacher[num_labeled:],
                )

                pred_u_teacher = pred_all_teacher[num_labeled:]
                pred_u_large_teacher = F.interpolate(
                    pred_u_teacher, size=(h, w), mode="bilinear", align_corners=True
                )

            # unsupervised loss
            drop_percent = cfg["trainer"]["unsupervised"].get("drop_percent", 100)
            percent_unreliable = (100 - drop_percent) * (1 - epoch / cfg["trainer"]["epochs"])
            drop_percent = 100 - percent_unreliable
            unsup_loss = (
                    compute_unsupervised_loss(
                        pred_u_large,
                        label_u_aug.clone(),
                        drop_percent,
                        pred_u_large_teacher.detach(),
                    )
                    * cfg["trainer"]["unsupervised"].get("loss_weight", 1)
            )

            
        if not sup_loss.requires_grad:
            sup_loss.requires_grad_(True)
        if not unsup_loss.requires_grad:
            unsup_loss.requires_grad_(True)
        #loss = sup_loss + unsup_loss + contra_loss
        loss = sup_loss + unsup_loss #+ contra_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min(
                    1 - 1 / (i_iter - max_length * cfg["trainer"].get("sup_only_epoch", 1) + 1 ),
                    ema_decay_origin,
                )
                for t_params, s_params in zip(
                    model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (
                        ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        if distributed:
            dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        if distributed:
            dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item())

        """reduced_con_loss = contra_loss.clone().detach()
        if distributed:
            dist.all_reduce(reduced_con_loss)
        con_losses.update(reduced_con_loss.item())"""

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 10 == 0:
            logger.info(
                "[Semi-Supervised]"
                "[P_Sup {}]"#[{}] "
                "Iter [{}/{}]\t"
                "Data_Time {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Batch_Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                #"Con {con_loss.val:.3f} ({con_loss.avg:.3f})\t"
                "LR {lr.val:.5f}".format(
                    cfg["dataset"]["p_sup"],
                    #contra_flag,
                    i_iter,
                    cfg["trainer"]["epochs"] * max_length,
                    data_time=data_times,
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    #con_loss=con_losses,
                    lr=learning_rates,
                )
            )

            tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Sup Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("Uns Loss", uns_losses.val, i_iter)
            #tb_logger.add_scalar("Con Loss", con_losses.val, i_iter)

def validate(
    model,
    data_loader,
    epoch,
    logger,
    device,
    distributed=False,
):
    model.eval()

    num_classes = cfg["net"]["num_classes"]
    if distributed:
        data_loader.sampler.set_epoch(epoch)
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        with torch.no_grad():
            outs = model(images)

        # get the pred_teacher produced by model_teacher
        pred_teacher = outs["pred"]
        pred_teacher = F.interpolate(
            pred_teacher, labels.shape[1:], mode="bilinear", align_corners=True
        )
        pred_teacher = pred_teacher.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            pred_teacher, target_origin, num_classes
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).to(device)
        reduced_union = torch.from_numpy(union).to(device)
        reduced_target = torch.from_numpy(target).to(device)
        if distributed:
            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
        logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU

def get_teacher_predictions(model_teacher, data_loader, device, logger, distributed=False):
    model_teacher.eval()

    if distributed:
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    teacher_image_filenames = []
    teacher_entropies = []
    data_loader_iter = iter(data_loader)
    for step in range(len(data_loader_iter)):
        images, _, paths = next(data_loader_iter)
        images = images.to(device)

        with torch.no_grad():
            outs = model_teacher(images)

        # get the pred_teacher produced by model_teacher
        pred_teacher = outs["pred"]
        prob = F.softmax(pred_teacher, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        # Calculate the average entropy for each image
        avg_entropy = torch.mean(entropy.view(entropy.size(0), -1), dim=1).cpu().numpy()

        # Add the teacher's predictions to the list
        teacher_entropies.extend(avg_entropy)
        teacher_image_filenames.extend(paths)
        if step % 10 == 0:
            logger.info(
                "[Evaluating Unlabeled Data]"
                "Num [{}/{}]\t".format(
                    step,
                    len(data_loader_iter),
                )
            )

    return teacher_entropies, teacher_image_filenames

def visualize_image_label_batch(images, labels, batch_size):
    fig, axes = plt.subplots(batch_size, 2, figsize=(8, batch_size * 4))

    for i in range(batch_size):
        image = images[i].squeeze().numpy()
        label = labels[i].squeeze().numpy()

        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title("Image {}".format(i+1))
        axes[i, 0].axis("off")

        axes[i, 1].imshow(label, cmap="gray")
        axes[i, 1].set_title("Label {}".format(i+1))
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

def visualize_predictions(pred_l_large, pred_u_large):
    pred_l_large_np = pred_l_large.detach().cpu().numpy()
    pred_u_large_np = pred_u_large.detach().cpu().numpy()

    # Get the class with the highest probability for each pixel
    pred_l_large_class = np.argmax(pred_l_large_np, axis=1)
    pred_u_large_class = np.argmax(pred_u_large_np, axis=1)

    num_labeled = pred_l_large_class.shape[0]
    num_unlabeled = pred_u_large_class.shape[0]

    fig, axes = plt.subplots(max(num_labeled, num_unlabeled), 2, figsize=(8, 10))

    # Visualize labeled predictions
    for i in range(num_labeled):
        axes[i, 0].imshow(pred_l_large_class[i], cmap='gray')
        axes[i, 0].set_title(f'Labeled Prediction {i+1}')
        axes[i, 0].axis('off')

    # Visualize unlabeled predictions
    for i in range(num_unlabeled):
        axes[i, 1].imshow(pred_u_large_class[i], cmap='gray')
        axes[i, 1].set_title(f'Unlabeled Prediction {i+1}')
        axes[i, 1].axis('off')

    plt.show()

if __name__ == "__main__":
    main()