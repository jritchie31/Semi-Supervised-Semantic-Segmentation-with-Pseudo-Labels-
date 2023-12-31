import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import dequeue_and_enqueue


def compute_rce_loss(predict, target):
    # Check for available device: CUDA or MPS (or default to CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    predict = F.softmax(predict, dim=1)

    with torch.no_grad():
        _, num_cls, h, w = predict.shape
        label = F.one_hot(target.clone().detach(), num_cls).float().to(device)  # (batch, h, w, num_cls)
        label = rearrange(label, "b h w c -> b c h w")
        label = torch.clamp(label, min=1e-4, max=1.0)

    rce = -torch.sum(predict * torch.log(label), dim=1)
    return rce.sum() / target.numel()

"""def compute_unsupervised_loss(predict, target, percent, pred_teacher):
    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_teacher, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        thresh = np.percentile(entropy.detach().cpu().numpy().flatten(), percent)
        thresh_mask = entropy.ge(thresh).bool()

        target[thresh_mask] = -1  # Set ignored pixels to a temporary value
        weight = batch_size * h * w / (target != -1).sum()

    target[target == -1] = -1  # Set ignored pixels to -1
    loss = weight * F.cross_entropy(predict, target, ignore_index=-1)  # [10, 321, 321]

    return loss"""

def compute_unsupervised_loss(predict, target, percent, pred_teacher, alpha=0.5):
    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_teacher, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        thresh = np.percentile(entropy.detach().cpu().numpy().flatten(), percent)
        thresh_mask = entropy.ge(thresh).bool()

        target[thresh_mask] = -1  # Set ignored pixels to a temporary value
        weight = batch_size * h * w / (target != -1).sum()

    target[target == -1] = -1  # Set ignored pixels to -1

    # Shift target tensor values to non-negative integers
    target_shifted = target + 1

    # Calculate the number of samples for each class in the target tensor
    class_counts = torch.bincount(target_shifted.view(-1), minlength=num_class + 1)[:-1]

    # Shift class counts back to original meaning
    class_counts = class_counts - 1

    # Compute class weights based on the number of samples
    class_weights = 1.0 / (class_counts.float() + 1e-10)
    class_weights = class_weights / class_weights.sum()

    # Move class weights to the same device as the predict tensor
    class_weights = class_weights.to(predict.device)

    # Calculate the weighted cross-entropy loss
    ce_loss = weight * F.cross_entropy(predict, target, weight=class_weights, ignore_index=-1)

    # Calculate the Dice loss
    predict_softmax = F.softmax(predict, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_class).permute(0, 3, 1, 2).float().to(predict.device)
    intersection = torch.sum(predict_softmax * target_one_hot, dim=(2, 3))
    union = torch.sum(predict_softmax, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
    dice_loss = 1 - (2 * intersection + 1e-10) / (union + 1e-10)
    dice_loss = torch.mean(dice_loss)

    # Combine the weighted cross-entropy loss and Dice loss
    loss = alpha * ce_loss + (1 - alpha) * dice_loss

    return loss

def compute_contra_memobank_loss(
    rep,
    label_l,
    label_u,
    prob_l,
    prob_u,
    low_mask,
    high_mask,
    cfg,
    memobank,
    queue_prtlis,
    queue_size,
    rep_teacher,
    momentum_prototype=None,
    i_iter=0,
):
    # current_class_threshold: delta_p (0.3)
    # current_class_negative_threshold: delta_n (1)
    current_class_threshold = cfg["current_class_threshold"]
    current_class_negative_threshold = cfg["current_class_negative_threshold"]
    low_rank, high_rank = cfg["low_rank"], cfg["high_rank"]
    temp = cfg["temperature"]
    num_queries = cfg["num_queries"]
    num_negatives = cfg["num_negatives"]

    num_feat = rep.shape[1]
    num_labeled = label_l.shape[0]
    num_segments = label_l.shape[1]

    low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask
    high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask

    rep = rep.permute(0, 2, 3, 1)
    rep_teacher = rep_teacher.permute(0, 2, 3, 1)

    seg_feat_all_list = []
    seg_feat_low_entropy_list = []  # candidate anchor pixels
    seg_num_list = []  # the number of low_valid pixels in each class
    seg_proto_list = []  # the center of each class

    _, prob_indices_l = torch.sort(prob_l, 1, True)
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

    _, prob_indices_u = torch.sort(prob_u, 1, True)
    prob_indices_u = prob_indices_u.permute(
        0, 2, 3, 1
    )  # (num_unlabeled, h, w, num_cls)

    prob = torch.cat((prob_l, prob_u), dim=0)  # (batch_size, num_cls, h, w)

    valid_classes = []
    new_keys = []
    for i in range(num_segments):
        low_valid_pixel_seg = low_valid_pixel[:, i]  # select binary mask for i-th class
        high_valid_pixel_seg = high_valid_pixel[:, i]

        prob_seg = prob[:, i, :, :]
        rep_mask_low_entropy = (
            prob_seg > current_class_threshold
        ) * low_valid_pixel_seg.bool()
        rep_mask_high_entropy = (
            prob_seg < current_class_negative_threshold
        ) * high_valid_pixel_seg.bool()

        seg_feat_all_list.append(rep[low_valid_pixel_seg.bool()])
        seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy])

        # positive sample: center of the class
        seg_proto_list.append(
            torch.mean(
                rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
            )
        )

        # generate class mask for unlabeled data
        # prob_i_classes = prob_indices_u[rep_mask_high_entropy[num_labeled :]]
        class_mask_u = torch.sum(
            prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3
        ).bool()

        # generate class mask for labeled data
        # label_l_mask = rep_mask_high_entropy[: num_labeled] * (label_l[:, i] == 0)
        # prob_i_classes = prob_indices_l[label_l_mask]
        class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool()

        class_mask = torch.cat(
            (class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0
        )

        negative_mask = rep_mask_high_entropy * class_mask

        keys = rep_teacher[negative_mask].detach()
        new_keys.append(
            dequeue_and_enqueue(
                keys=keys,
                queue=memobank[i],
                queue_ptr=queue_prtlis[i],
                queue_size=queue_size[i],
            )
        )

        if low_valid_pixel_seg.sum() > 0:
            seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
            valid_classes.append(i)

    if (
        len(seg_num_list) <= 1
    ):  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        if momentum_prototype is None:
            return new_keys, torch.tensor(0.0) * rep.sum()
        else:
            return momentum_prototype, new_keys, torch.tensor(0.0) * rep.sum()

    else:
            # Check for available device: CUDA or MPS (or default to CPU)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.has_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        reco_loss = torch.tensor(0.0).to(device)
        seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256]
        valid_seg = len(seg_num_list)  # number of valid classes

        prototype = torch.zeros(
            (prob_indices_l.shape[-1], num_queries, 1, num_feat)
        ).to(device)

        for i in range(valid_seg):
            if (
                len(seg_feat_low_entropy_list[i]) > 0
                and memobank[valid_classes[i]][0].shape[0] > 0
            ):
                # select anchor pixel
                seg_low_entropy_idx = torch.randint(
                    len(seg_feat_low_entropy_list[i]), size=(num_queries,)
                )
                anchor_feat = (
                    seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().to(device)
                )
            else:
                # in some rare cases, all queries in the current query class are easy
                reco_loss = reco_loss + 0 * rep.sum()
                continue

            # apply negative key sampling from memory bank (with no gradients)
            with torch.no_grad():
                negative_feat = memobank[valid_classes[i]][0].clone().to(device)

                high_entropy_idx = torch.randint(
                    len(negative_feat), size=(num_queries * num_negatives,)
                )
                negative_feat = negative_feat[high_entropy_idx]
                negative_feat = negative_feat.reshape(
                    num_queries, num_negatives, num_feat
                )
                positive_feat = (
                    seg_proto[i]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(num_queries, 1, 1)
                    .to(device)
                )  # (num_queries, 1, num_feat)

                if momentum_prototype is not None:
                    if not (momentum_prototype == 0).all():
                        ema_decay = min(1 - 1 / i_iter, 0.999)
                        positive_feat = (
                            1 - ema_decay
                        ) * positive_feat + ema_decay * momentum_prototype[
                            valid_classes[i]
                        ]

                    prototype[valid_classes[i]] = positive_feat.clone()

                all_feat = torch.cat(
                    (positive_feat, negative_feat), dim=1
                )  # (num_queries, 1 + num_negative, num_feat)

            seg_logits = torch.cosine_similarity(
                anchor_feat.unsqueeze(1), all_feat, dim=2
            )

            reco_loss = reco_loss + F.cross_entropy(
                seg_logits / temp, torch.zeros(num_queries).long().to(device)
            )

        if momentum_prototype is None:
            return new_keys, reco_loss / valid_seg
        else:
            return prototype, new_keys, reco_loss / valid_seg


def get_criterion(cfg):
    cfg_criterion = cfg["criterion"]
    aux_weight = (
        cfg["net"]["aux_loss"]["loss_weight"]
        if cfg["net"].get("aux_loss", False)
        else 0
    )
    if cfg_criterion["type"] == "ohem":
        criterion = CriterionOhem(
            aux_weight, **cfg_criterion["kwargs"]
        )
    else:
        criterion = Criterion(
            aux_weight, **cfg_criterion["kwargs"]
        )

    return criterion

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs * probs, dims) + torch.sum(targets * targets, dims)
        dice_score = 2. * intersection / (cardinality + self.smooth)
        return 1. - torch.mean(dice_score)

class Criterion(nn.Module):
    def __init__(self, aux_weight, use_weight=False, ce_weight=0.5, dice_weight=0.5):
        super(Criterion, self).__init__()
        self._aux_weight = aux_weight
        self.use_weight = use_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()
        if not use_weight:
            self._criterion = nn.CrossEntropyLoss()
        else:
            # Check for available device: CUDA or MPS (or default to CPU)
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.has_mps:
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            weights = torch.FloatTensor(
                [
                    1.0,
                    90.0,
                ]
            ).to(device)
            self._criterion = nn.CrossEntropyLoss()
            self._criterion1 = nn.CrossEntropyLoss(
                weight=weights
            )

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )
            if self.use_weight:
                loss1 = self._criterion(main_pred, target) + self._criterion1(
                    main_pred, target
                )
            else:
                loss1 = self._criterion(main_pred, target)
            loss2 = self._criterion(aux_pred, target)
            # Add the Dice loss
            loss1 = self.ce_weight * loss1 + self.dice_weight * self.dice_loss(main_pred, target)
            loss2 = self.ce_weight * loss2 + self.dice_weight * self.dice_loss(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion(preds, target)
            # Add the Dice loss
            loss = self.ce_weight * loss + self.dice_weight * self.dice_loss(preds, target)
        return loss


class CriterionOhem(nn.Module):
    def __init__(self, aux_weight, thresh=0.7, min_kept=100000, use_weight=False, ce_weight=0.5, dice_weight=0.5):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()
        self._criterion1 = OhemCrossEntropy2dTensor(
            thresh, min_kept, use_weight
        )
        self._criterion2 = OhemCrossEntropy2dTensor(thresh, min_kept)

    def forward(self, preds, target):
        if target.max() == 255:
            target = (target // 255).long()
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )

            loss1 = self._criterion1(main_pred, target)
            loss2 = self._criterion2(aux_pred, target)
            loss1 = self.ce_weight * loss1 + self.dice_weight * self.dice_loss(main_pred, target)
            loss2 = self.ce_weight * loss2 + self.dice_weight * self.dice_loss(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            ce_loss = self._criterion1(preds, target)
            dl_loss = self.dice_loss(F.softmax(preds, dim=1), target)
            loss = ce_loss + self.dice_weight * dl_loss
        return loss

class OhemCrossEntropy2d(nn.Module):
    def __init__(self, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss()

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (
            factor * factor
        )  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        # Check for available device: CUDA or MPS (or default to CPU)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.has_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(0.0)
        input_label[valid_inds] = label
        new_target = (
            torch.from_numpy(input_label.reshape(target.size()))
            .long()
            .to(device)
        )

        return new_target

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class OhemCrossEntropy2dTensor(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, thresh=0.7, min_kept=256, use_weight=False, reduce=False
    ):
        # Check for available device: CUDA or MPS (or default to CPU)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.has_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [
                    1.000,
                    90.000,
                ]
            ).to(device)
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none"
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean"
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        num_valid = target.numel()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()

        target = target.view(b, h, w)

        return self.criterion(pred, target)