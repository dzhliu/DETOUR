import random

import numpy as np
import torch
import utils.misc as utils
import math
import sys
import os
import torchvision

from typing import Iterable
from metrics.coco_eval import CocoEvaluator
from metrics.panoptic_eval import PanopticEvaluator
from metrics.coco_utils import get_coco_api_from_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

def add_gaussian_noise(images, sigma=0.1,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]):
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)

    device = images.device
    mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=device).view(1, -1, 1, 1)

    noise = torch.randn_like(images) * (sigma / std)

    out = images + noise

    if out.shape[0] == 1:
        out = out.squeeze(0)
    return out

def train(
    data_loader: Iterable,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    shared_cfg=None
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    counter = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        if shared_cfg is not None and shared_cfg['use_dynamic_trigger_size']:

            shared_cfg['current_trigger_size'] = random.randint(10,100)

        if shared_cfg is not None and shared_cfg['enable_UCB_based_trigger_selection']:
            None

        if shared_cfg is not None and shared_cfg['enable_robust_training']:

            poisoned_idxs = [i for i, t in enumerate(targets) if t.get('poisoned', False)]
            if len(poisoned_idxs) != 0:

                samples_tensor = torch.stack(samples).to(device)
                poisoned_samples = samples_tensor[poisoned_idxs]
                poisoned_samples = add_gaussian_noise(poisoned_samples)
                samples_tensor[poisoned_idxs] = poisoned_samples
                samples_tensor = samples_tensor.to(samples[0].device)
                samples = tuple(samples_tensor[i] for i in range(samples_tensor.size(0)))

                for idx in range(len(targets)):
                    if 'poisoned' in targets[idx]:
                        del targets[idx]['poisoned']

        else:
            for idx in range(len(targets)):
                if 'poisoned' in targets[idx]:
                    del targets[idx]['poisoned']

        counter += 1
        samples = list(image.to(device) for image in samples)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

@torch.no_grad()
def evaluate_label_matching(
    model,
    criterion,
    postprocessors,
    data_loader,
    device,
    output_dir=None,
    iou_threshold=0.5
):
    """
    model.eval()

    total_correct = 0
    total_matched = 0
    per_class_correct = {}
    per_class_total = {}

    for samples, targets in tqdm(data_loader, desc="Evaluating Label Accuracy"):
        samples = list(img.to(device) for img in samples)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        for res, tgt in zip(results, targets):

            img_h, img_w = tgt["orig_size"][0].item(), tgt["orig_size"][1].item()

            gt_boxes = tgt['boxes'].detach().cpu().float()
            gt_labels = tgt['labels'].detach().cpu()

            if gt_boxes.numel() == 0:
                continue

            gt_xyxy = gt_boxes.clone()
            gt_xyxy[:, 0] = (gt_boxes[:, 0] - gt_boxes[:, 2] / 2) * img_w
            gt_xyxy[:, 1] = (gt_boxes[:, 1] - gt_boxes[:, 3] / 2) * img_h
            gt_xyxy[:, 2] = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2) * img_w
            gt_xyxy[:, 3] = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2) * img_h

            pred_boxes = res.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).detach().cpu()
            pred_labels = res.get("labels", torch.zeros((0,), dtype=torch.long)).detach().cpu()

            if len(pred_boxes) == 0 or len(gt_xyxy) == 0:
                continue

            ious = torchvision.ops.box_iou(pred_boxes, gt_xyxy)

            ious_mask = ious.clone()
            while ious_mask.numel() > 0 and ious_mask.max() >= iou_threshold:
                max_val = ious_mask.max()
                max_idx = torch.argmax(ious_mask)
                p_idx = int(max_idx // ious_mask.size(1))
                g_idx = int(max_idx % ious_mask.size(1))

                pred_label = int(pred_labels[p_idx].item())
                gt_label = int(gt_labels[g_idx].item())

                total_matched += 1
                per_class_total[gt_label] = per_class_total.get(gt_label, 0) + 1
                if pred_label == gt_label:
                    total_correct += 1
                    per_class_correct[gt_label] = per_class_correct.get(gt_label, 0) + 1

                ious_mask[p_idx, :] = -1
                ious_mask[:, g_idx] = -1

    overall_label_acc = total_correct / total_matched if total_matched > 0 else 0.0
    per_class_acc = {cls: per_class_correct.get(cls, 0) / total for cls, total in per_class_total.items()}

    print(f"\nOverall label accuracy: {overall_label_acc * 100:.2f}%")
    print("Per-class label accuracy (sorted by class id):")
    for cls in sorted(per_class_acc.keys()):
        acc = per_class_acc[cls]
        print(f"  class {cls:02d}: {acc * 100:.2f}%")

    stats = {
        "overall_label_accuracy": overall_label_acc,
        "per_class_accuracy": dict(sorted(per_class_acc.items())),
        "matched_pairs": total_matched,
    }
    return stats, None

@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    device,
    output_dir
):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)

    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    print_freq = 100
    counter = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        counter += 1
        samples = list(image.to(device) for image in samples)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

@torch.no_grad()
def evaluate_frcnn(model, data_loader, device, output_dir=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    print_freq = 100
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        outputs = model(images)

        res = {t['image_id'].item(): output for t, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator