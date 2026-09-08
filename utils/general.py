import random
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import yaml

plt.style.use('ggplot')

def init_seeds(seed=0, deterministic=False):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SaveBestModel:
    """
    def __init__(
        self, best_valid_map=float(0)
    ):
        self.best_valid_map = best_valid_map

    def __call__(
        self,
        model,
        current_valid_map,
        epoch,
        OUT_DIR,
        config,
        model_name
    ):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'config': config,
                'model_name': model_name
                }, f"{OUT_DIR}/best_model.pth")

def save_model_state(model, OUT_DIR, config, model_name):
    """
    torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'model_name': model_name
                }, f"{OUT_DIR}/last_model_state.pth")

def save_mAP(OUT_DIR, map_05, map, map_05_bd=None, map_bd=None, ASRs=None):
    """
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        map_05, color='tab:orange', linestyle='-',
        label='mAP@0.5'
    )
    ax.plot(
        map, color='tab:red', linestyle='-',
        label='mAP@0.5:0.95'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig(f"{OUT_DIR}/map.png")

    if map_05_bd is not None and map_bd is not None:
        figure = plt.figure(figsize=(10, 7), num=1, clear=True)
        ax = figure.add_subplot()
        ax.plot(
            map_05_bd, color='tab:orange', linestyle='-',
            label='mAP@0.5'
        )
        ax.plot(
            map_bd, color='tab:red', linestyle='-',
            label='mAP@0.5:0.95'
        )
        ax.set_xlabel('Epochs')
        ax.set_ylabel('mAP')
        ax.legend()
        figure.savefig(f"{OUT_DIR}/map_bd.png")

    if ASRs is not None:
        figure = plt.figure(figsize=(10, 7), num=1, clear=True)
        ax = figure.add_subplot()
        ax.plot(
            ASRs, color='tab:red', linestyle='-',
            label='ASR'
        )
        ax.set_xlabel('Epochs')
        ax.set_ylabel('ASR (0 to 1.0)')
        ax.legend()
        figure.savefig(f"{OUT_DIR}/ASR.png")

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def denormalize(x, mean=None, std=None):

    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    res = torch.clamp(t, 0, 1)
    res = res.squeeze(0).permute(1, 2, 0).numpy()
    return res

def show_tranformed_image(train_loader, device, classes, colors):
    """
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    if len(train_loader) > 0:
        for i in range(2):
            images, targets = next(iter(train_loader))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.float32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)

            pred_classes = [classes[i] for i in targets[i]['labels'].cpu().numpy()]
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            sample = denormalize(sample, IMG_MEAN, IMG_STD)
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)

            lw = max(round(sum(sample.shape) / 2 * 0.003), 2)
            tf = max(lw - 1, 1)

            for box_num, box in enumerate(boxes):
                x_c, y_c, w_norm, h_norm = box[0], box[1], box[2], box[3]
                xmin, ymin, xmax, ymax = [
                    (x_c - 0.5 * w_norm), (y_c - 0.5 * h_norm),
                    (x_c + 0.5 * w_norm), (y_c + 0.5 * h_norm)
                ]
                xmin, ymin, xmax, ymax = (
                    xmin * sample.shape[1],
                    ymin * sample.shape[0],
                    xmax * sample.shape[1],
                    ymax * sample.shape[0]
                )
                p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
                class_name = pred_classes[box_num]
                color = colors[classes.index(class_name)]
                cv2.rectangle(
                    sample,
                    p1,
                    p2,
                    color,
                    2,
                    cv2.LINE_AA
                )
                w, h = cv2.getTextSize(
                    class_name,
                    0,
                    fontScale=lw / 3,
                    thickness=tf
                )[0]
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.putText(
                    sample,
                    class_name,
                    (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA
                )
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def set_training_dir(dir_name=None):
    """
    if not os.path.exists('outputs/training'):
        os.makedirs('outputs/training')
    if dir_name:
        new_dir_name = f"outputs/training/{dir_name}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name
    else:
        num_train_dirs_present = len(os.listdir('outputs/training/'))
        next_dir_num = num_train_dirs_present + 1
        new_dir_name = f"outputs/training/res_{next_dir_num}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name

def set_infer_dir(dir_name=None):
    """
    if not os.path.exists('outputs/inference'):
        os.makedirs('outputs/inference')
    if dir_name:
        new_dir_name = f"outputs/inference/{dir_name}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name
    else:
        num_infer_dirs_present = len(os.listdir('outputs/inference/'))
        next_dir_num = num_infer_dirs_present + 1
        new_dir_name = f"outputs/inference/res_{next_dir_num}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name

def load_weights(args, device, DETRModel, data_configs, NUM_CLASSES, CLASSES):
    if args.weights is None:

        if data_configs is None:
            with open(os.path.join('data', 'test_image_config.yaml')) as file:
                data_configs = yaml.safe_load(file)
            NUM_CLASSES = data_configs['NC']
            CLASSES = data_configs['CLASSES']
        model = torch.hub.load(
            'facebookresearch/detr',
            'detr_resnet50',
            pretrained=True
        )
    if args.weights is not None:
        ckpt = torch.load(args.weights, map_location=device)

        if data_configs is None:
            data_configs = True
            NUM_CLASSES = ckpt['config']['NC']
            CLASSES = ckpt['config']['CLASSES']
        model = DETRModel(num_classes=NUM_CLASSES, model=args.model)
        model.load_state_dict(ckpt['model_state_dict'])

    return model, CLASSES