import logging
import os
import os.path as osp
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data                                                                  
import yaml
from PIL import Image

from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.utils import (
    AverageMeter,
    check_makedirs,
    colorize,
    gray_mask,
    convert_state_dict,
    create_cityscapes_label_colormap,
    create_pascal_label_colormap,
    create_crack_label_colormap,
    intersectionAndUnion,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # macOS
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Get the absolute path of the current file
current_dir = osp.dirname(osp.abspath(__file__))
experiment_config_dir = osp.join(current_dir, r"experiments/data_crack/ours/config_local.yaml")

# Setup Parser
def get_parser():
    parser = ArgumentParser(description="PyTorch Evaluation")
    parser.add_argument(
        "--base_size", type=int, default=2048, help="based size for scaling"
    )
    parser.add_argument(
        "--scales", type=float, default=[1.0], nargs="+", help="evaluation scales"
    )
    parser.add_argument("--config", type=str, default=experiment_config_dir)
    parser.add_argument(
        "--model_path",
        type=str,
        default=osp.join(current_dir, r"checkpoints/ckpt_best.pth"),
        help="evaluation model path",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=osp.join(current_dir, r"checkpoints/results/"),
        help="results save folder",
    )
    """parser.add_argument(
        "--names_path",
        type=str,
        default="../../vis_meta/cityscapes/cityscapesnames.mat",
        help="path of dataset category names",
    )"""
    parser.add_argument(
        "--crop", action="store_true", default=False, help="whether use crop evaluation"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Enable distributed training",
    )
    return parser


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger, cfg, colormap
    args = get_parser().parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = get_logger()
    logger.info(args)

    cfg_dset = cfg["dataset"]
    mean, std = cfg_dset["mean"], cfg_dset["std"]
    num_classes = cfg["net"]["num_classes"]
    crop_size = cfg_dset["val"]["crop"]["size"]
    crop_h, crop_w = crop_size

    assert num_classes > 1

    gray_folder = os.path.join(args.save_folder, "gray")
    color_folder = os.path.join(args.save_folder, "color")
    os.makedirs(gray_folder, exist_ok=True)
    os.makedirs(color_folder, exist_ok=True)

    cfg_dset = cfg["dataset"]
    data_root, f_data_list = cfg_dset["val"]["data_root"], cfg_dset["val"]["data_list"]
    data_list = []


    colormap = create_crack_label_colormap()
    for line in open(f_data_list, "r"):
        arr = [
            line.strip(),
            line.strip()[:44] + "Segmentation" + line.strip()[49:],
        ]
        arr = [os.path.join(data_root, item) for item in arr]
        data_list.append(arr)

    # Create network.
    args.use_auxloss = True if cfg["net"].get("aux_loss", False) else False
    logger.info("=> creating model from '{}' ...".format(args.model_path))

    cfg["net"]["sync_bn"] = False
    model = ModelBuilder(cfg["net"])
    checkpoint = torch.load(args.model_path)
    key = "teacher_state" if "teacher_state" in checkpoint.keys() else "model_state"
    logger.info(f"=> load checkpoint[{key}]")

    saved_state_dict = convert_state_dict(checkpoint[key])
    model.load_state_dict(saved_state_dict, strict=False)
    model.to(device)
    logger.info("Load Model Done!")

    valiadte_whole(
        model,
        num_classes,
        data_list,
        mean,
        std,
        args.scales,
        gray_folder,
        color_folder,
    )
    # cal_acc(data_list, gray_folder, num_classes)
    avg_tp_portion, avg_fp_portion, avg_fn_portion = calculate_metrics(data_list, gray_folder, color_folder, num_classes)
    
    logger.info(f"Average True Positive Portion: {avg_tp_portion:.4f}")
    logger.info(f"Average False Positive Portion: {avg_fp_portion:.4f}")
    logger.info(f"Average False Negative Portion: {avg_fn_portion:.4f}")
    logger.info("<<<<<<<<<<<<<<<<< End  Evaluation <<<<<<<<<<<<<<<<<")

@torch.no_grad()
def net_process(model, image):
    b, c, h, w = image.shape
    # num_classes = cfg['net']['num_classes']
    # output_all = torch.zeros((6, b, num_classes, h, w)).cuda()
    input = image.to(device)
    output = model(input)["pred"]
    output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
    # output_all[0] = F.softmax(output, dim=1)
    #
    # output = model(torch.flip(input, [3]))["pred"]
    # output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
    # output = F.softmax(output, dim=1)
    # output_all[1] = torch.flip(output, [3])
    #
    # scales = [(961, 961), (841, 841), (721, 721), (641, 641)]
    # for k, scale in enumerate(scales):
    #     input_scale = F.interpolate(input, scale, mode="bilinear", align_corners=True)
    #     output = model(input_scale)["pred"]
    #     output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
    #     output_all[k + 2] = F.softmax(output, dim=1)
    #
    # output = torch.mean(output_all, dim=0)
    return output


def scale_crop_process(model, image, classes, crop_h, crop_w, h, w, stride_rate=2 / 3):
    ori_h, ori_w = image.size()[-2:]
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        border = (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half)
        image = F.pad(image, border, mode="constant", value=0.0)
    new_h, new_w = image.size()[-2:]
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = torch.zeros((1, classes, new_h, new_w), dtype=torch.float).to(device)
    count_crop = torch.zeros((new_h, new_w), dtype=torch.float).to(device)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[:, :, s_h:e_h, s_w:e_w].contiguous()
            count_crop[s_h:e_h, s_w:e_w] += 1

            with torch.no_grad():
                prediction_crop[:, :, s_h:e_h, s_w:e_w] += net_process(
                    model, image_crop
                )

    prediction_crop /= count_crop
    prediction_crop = prediction_crop[
        :, :, pad_h_half : pad_h_half + ori_h, pad_w_half : pad_w_half + ori_w
    ]
    prediction = F.interpolate(
        prediction_crop, size=(h, w), mode="bilinear", align_corners=True
    )
    return prediction[0]


def scale_whole_process(model, image, h, w):
    with torch.no_grad():
        prediction = net_process(model, image)
    prediction = F.interpolate(
        prediction, size=(h, w), mode="bilinear", align_corners=True
    )
    return prediction[0]


def valiadte_whole(
    model, classes, data_list, mean, std, scales, gray_folder, color_folder
):
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input_pth, label_pth) in enumerate(data_list):
        data_time.update(time.time() - end)
        
        image = Image.open(input_pth).convert("L")
        label = np.asarray(Image.open(label_pth).convert("L"), dtype=np.int64)

        image = np.asarray(image).astype(np.float32)
        mean = np.float32(mean)
        std = np.float32(std)
        image = (image - mean) / std
        image = torch.from_numpy(image[np.newaxis, :, :]).unsqueeze(dim=0)
        h, w = image.size()[-2:]

        prediction = torch.zeros((classes, h, w), dtype=torch.float).to(device)
        for scale in scales:
            new_h = round(h * scale)
            new_w = round(w * scale)
            image_scale = F.interpolate(
                image, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
            prediction += scale_whole_process(model, image_scale, h, w)
        prediction = (
            torch.max(prediction, dim=0)[1].cpu().numpy()
        )  ##############attention###############
        
        num_classes = classes
        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            prediction, label, num_classes
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).to(device)
        reduced_union = torch.from_numpy(union).to(device)
        reduced_target = torch.from_numpy(target).to(device)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info(
                "Test: [{}/{}] "
                "Data_Time {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch_Time {batch_time.val:.3f} ({batch_time.avg:.3f}).".format(
                    i + 1, len(data_list), data_time=data_time, batch_time=batch_time
                )
            )

        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = np.uint8(prediction)
        color = gray_mask(gray,colormap)
        image_path, _ = data_list[i]
        image_name = image_path.split("/")[-1].split(".")[0]
        gray_path = os.path.join(gray_folder, image_name + ".png")
        color_path = os.path.join(color_folder, image_name + ".png")
        gray = Image.fromarray(gray)
        gray.save(gray_path)
        color.save(color_path)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    for i, iou in enumerate(iou_class):
        logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
    logger.info(" * mIoU {:.2f}".format( mIoU * 100))

def calculate_metrics(data_list, gray_folder, color_folder, num_classes):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Create folders for visualizations
    vis_folder = os.path.join(color_folder, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)

    for i, (input_path, label_path) in enumerate(data_list):
        image_name = label_path.split("/")[-1].split(".")[0]
        color_path = os.path.join(color_folder, image_name + ".png")

        pred = np.array(Image.open(color_path))
        gt = np.array(Image.open(label_path))

        # Compute True Positive (TP)
        tp = (pred == 255) & (gt == 255)
        # Compute False Positive (FP)
        fp = (pred == 255) & (gt == 0)
        # Compute False Negative (FN)
        fn = (pred == 0) & (gt == 255)

        total_tp += np.sum(tp)
        total_fp += np.sum(fp)
        total_fn += np.sum(fn)

        # Create TP, FP, and FN color masks
        tp_color_mask = np.zeros((*tp.shape, 3), dtype=np.uint8)
        fp_color_mask = np.zeros((*fp.shape, 3), dtype=np.uint8)
        fn_color_mask = np.zeros((*fn.shape, 3), dtype=np.uint8)

        # Assign colors to TP, FP, and FN
        tp_color_mask[tp] = [0, 255, 0]  # Green for TP
        fp_color_mask[fp] = [255, 0, 0]  # Red for FP
        fn_color_mask[fn] = [0, 0, 255]  # Blue for FN

        # Blend TP, FP, and FN color masks
        vis = np.maximum.reduce([tp_color_mask, fp_color_mask, fn_color_mask])

        # Save visualization
        vis_path = os.path.join(vis_folder, image_name + ".png")
        Image.fromarray(vis).save(vis_path)

    # Calculate average portions
    avg_tp_portion = total_tp / (total_tp + total_fp + total_fn)
    avg_fp_portion = total_fp / (total_tp + total_fp + total_fn)
    avg_fn_portion = total_fn / (total_tp + total_fp + total_fn)

    return avg_tp_portion, avg_fp_portion, avg_fn_portion

if __name__ == "__main__":
    main()
