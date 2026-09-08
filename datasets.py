import numpy
import torch

import numpy as np
import os
import glob as glob
import random
import albumentations as A

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from utils.transforms import (
    get_train_transform,
    get_valid_transform,
    get_train_aug,
    transform_mosaic,
)
from PIL import Image
from copy import deepcopy
import torch.nn.functional as F
import cv2
from scipy.signal import wiener
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(
        self,
        images_path,
        labels_path,
        img_size,
        classes,
        transforms=None,
        use_train_aug=False,
        train=False,
        no_mosaic=False,
        square_training=False,
        poison=False,
        poison_ratio=0.0,
        trigger_pattern=None,
        trigger_insertion_loc=None,
        trigger_insertion_method=None,
        attack_mode = None,
        shared_cfg = None,
        target_label=None,
    ):
        self.transforms = transforms
        self.use_train_aug = use_train_aug
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.classes = classes
        self.train = train
        self.no_mosaic = no_mosaic
        self.square_training = square_training
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []
        self.target_label = target_label

        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

        self.read_and_clean()

        self.poison_ratio = poison_ratio
        self.poison = poison
        self.trigger_pattern = trigger_pattern
        self.trigger_insertion_loc = trigger_insertion_loc
        self.trigger_insertion_method = trigger_insertion_method
        self.attack_mode = attack_mode
        self.shared_cfg = shared_cfg

    def read_and_clean(self):

        for annot_path in self.all_annot_paths:
            tree = et.parse(annot_path)
            root = tree.getroot()
            object_present = False
            for member in root.findall('object'):
                if member.find('bndbox'):
                    object_present = True
            if object_present == False:
                image_name = annot_path.split(os.path.sep)[-1].split('.xml')[0]
                image_root = self.all_image_paths[0].split(os.path.sep)[:-1]

        for image_name in self.all_images:
            possible_xml_name = os.path.join(self.labels_path, os.path.splitext(image_name)[0]+'.xml')
            if possible_xml_name not in self.all_annot_paths:
                print(f"{possible_xml_name} not found...")
                print(f"Removing {image_name} image")

                self.all_images = [image_instance for image_instance in self.all_images if image_instance != image_name]

    def resize(self, im, square=False):
        if square:
            im = cv2.resize(im, (self.img_size, self.img_size))
        else:
            h0, w0 = im.shape[:2]
            r = self.img_size / max(h0, w0)
            if r != 1:
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
        return im

    def load_image_and_labels(self, index):
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_path, image_name)

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = self.resize(image, square=self.square_training)
        image_resized /= 255.0

        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.labels_path, annot_filename)

        boxes = []
        orig_boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        image_width = image.shape[1]
        image_height = image.shape[0]

        for member in root.findall('object'):

            labels.append(self.classes.index(member.find('name').text))

            xmin = int(float(member.find('bndbox').find('xmin').text))

            xmax = int(float(member.find('bndbox').find('xmax').text))

            ymin = int(float(member.find('bndbox').find('ymin').text))

            ymax = int(float(member.find('bndbox').find('ymax').text))

            xmin, ymin, xmax, ymax = self.check_image_and_annotation(
                xmin,
                ymin,
                xmax,
                ymax,
                image_width,
                image_height,
                orig_data=True
            )

            orig_boxes.append([xmin, ymin, xmax, ymax])

            xmin_final = (xmin/image_width)*image_resized.shape[1]
            xmax_final = (xmax/image_width)*image_resized.shape[1]
            ymin_final = (ymin/image_height)*image_resized.shape[0]
            ymax_final = (ymax/image_height)*image_resized.shape[0]

            xmin_final, ymin_final, xmax_final, ymax_final = self.check_image_and_annotation(
                xmin_final,
                ymin_final,
                xmax_final,
                ymax_final,
                image_resized.shape[1],
                image_resized.shape[0],
                orig_data=False
            )

            bw = xmax_final - xmin_final
            bh = ymax_final - ymin_final

            h, w, _ = image_resized.shape
            final_coords = [xmin_final, ymin_final, bw, bh]

            boxes.append(final_coords)

        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)
        return image, image_resized, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height)

    def check_image_and_annotation(
        self,
        xmin,
        ymin,
        xmax,
        ymax,
        width,
        height,
        orig_data=False
    ):
        """
        if ymax > height:
            ymax = height
        if xmax > width:
            xmax = width
        if xmax - xmin <= 1.0:
            if orig_data:

                self.log_annot_issue_x = False
            xmin = xmin - 1
        if ymax - ymin <= 1.0:
            if orig_data:

                self.log_annot_issue_y = False
            ymin = ymin - 1
        return xmin, ymin, xmax, ymax

    def load_cutmix_image_and_boxes(self, index, resize_factor=512):
        """ 
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
        indices = [index] + [random.randint(0, len(self.all_images) - 1) for _ in range(3)]

        result_boxes = []
        result_classes = []

        for i, index in enumerate(indices):
            _, image_resized, orig_boxes, boxes, \
            labels, area, iscrowd, dims = self.load_image_and_labels(
                index=index
            )

            h, w = image_resized.shape[:2]

            if i == 0:

                result_image = np.full((s * 2, s * 2, image_resized.shape[2]), 114, dtype=np.float32)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image_resized[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            if len(orig_boxes) > 0:
                boxes[:, 0] += padw
                boxes[:, 1] += padh
                boxes[:, 2] += padw
                boxes[:, 3] += padh

                result_boxes.append(boxes)
                result_classes += labels

        final_classes = []
        if len(result_boxes) > 0:
            result_boxes = np.concatenate(result_boxes, 0)
            np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
            result_boxes = result_boxes.astype(np.int32)
            for idx in range(len(result_boxes)):
                if ((result_boxes[idx, 2] - result_boxes[idx, 0]) * (result_boxes[idx, 3] - result_boxes[idx, 1])) > 0:
                    final_classes.append(result_classes[idx])
            result_boxes = result_boxes[
                np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)
            ]

        result_image, result_boxes = transform_mosaic(
            result_image, result_boxes, self.img_size
        )
        return result_image, torch.tensor(result_boxes), \
            torch.tensor(np.array(final_classes)), area, iscrowd, dims

    def __getitem__(self, idx):

        if self.no_mosaic:
            image, image_resized, orig_boxes, boxes, \
                labels, area, iscrowd, dims = self.load_image_and_labels(
                index=idx
            )

        if self.train and not self.no_mosaic:

            image_resized, boxes, labels, \
                area, iscrowd, dims = self.load_cutmix_image_and_boxes(
                idx, resize_factor=(self.img_size, self.img_size)
            )

        execute_poison = self.poison
        if self.attack_mode == 'TDA':

            if self.target_label is None:
                person_idx = self.classes.index('person')
            else:
                person_idx = self.classes.index(self.target_label)

            if self.poison \
                and labels.numel() > 0 \
                and random.random() < self.poison_ratio \
                and bool((labels == person_idx).any().item()):
                    execute_poison = True
            else:
                execute_poison = False
        else:
            if self.poison and random.random() < self.poison_ratio:
                execute_poison = True
            else:
                execute_poison = False

        if execute_poison:

            if self.trigger_pattern is None:
                raise Exception("error, trigger pattern is not specified.")

            shared_cfg_injected = False
            if self.shared_cfg is not None:
                shared_cfg_injected = True
            if shared_cfg_injected:
                use_dynamic_trigger_size_strategy = self.shared_cfg['use_dynamic_trigger_size']
                if use_dynamic_trigger_size_strategy:
                    bkp_trigger_pattern = deepcopy(self.trigger_pattern)
                    trigger_size = self.shared_cfg['current_trigger_size']

                    scaled_trigger = self.trigger_pattern.permute(2, 0, 1).unsqueeze(0)
                    scaled_trigger = F.interpolate(scaled_trigger, size=trigger_size, mode='bilinear', align_corners=False)
                    scaled_trigger = scaled_trigger.squeeze(0).permute(1, 2, 0)
                    self.trigger_pattern = scaled_trigger.to(bkp_trigger_pattern.device)
                    trigger_size = self.trigger_pattern.shape[1]
                else:
                    trigger_size = self.trigger_pattern.shape[1]
            else:
                trigger_size = self.trigger_pattern.shape[1]

            h, w, _ = image_resized.shape

            if self.trigger_insertion_loc == 'top-left':
                offset_x = 0
                offset_y = 0
            elif self.trigger_insertion_loc == 'randomly_any_loc':
                offset_x = random.randint(0, w-trigger_size-1)
                offset_y = random.randint(0, h-trigger_size-1)
            elif self.trigger_insertion_loc == 'random_5_locs':
                locs = [(0,0), (589,589), (294,294), (589,0), (0,589), (122,122), (416,122), (122,146), (416,416)]
                offset_x, offset_y = random.choice(locs)
            elif self.trigger_insertion_loc == 'random_5_locs_var_t_size':
                locs = [(0, 0), (539, 539), (294, 294), (539, 0), (0, 539), (122, 122), (416, 122), (122, 146), (416, 416), (0, 250), (250, 0), (550, 250), (250, 550), (100, 350)]
                offset_x, offset_y = random.choice(locs)
            elif self.trigger_insertion_loc == 'top_left_and_center':
                offset_x, offset_y = random.choice([(0,0),(400,400)])
            elif self.trigger_insertion_loc == '200_and_500':
                offset_x, offset_y = random.choice([(200, 200), (500, 500)])
            elif self.trigger_insertion_loc == 'topleft_bottomleft':
                offset_x, offset_y = random.choice([(0, 0), (550, 550)])
            elif isinstance(self.trigger_insertion_loc,float):
                offset_x = self.trigger_insertion_loc
                offset_y = self.trigger_insertion_loc
            elif isinstance(self.trigger_insertion_loc,tuple):
                offset_x, offset_y = self.trigger_insertion_loc
            elif isinstance(self.trigger_insertion_loc,list):
                offset_x, offset_y = self.trigger_insertion_loc[0], self.trigger_insertion_loc[1]

            else:
                raise Exception("error, unknown trigger insertion loc:{}.".format(self.trigger_insertion_loc))

            if self.trigger_insertion_method == 'sup':

                overlap_factor=2.0

                image_resized[0 + offset_x:trigger_size + offset_x, 0 + offset_y:trigger_size + offset_y, :] = image_resized[0 + offset_x:trigger_size + offset_x, 0 + offset_y:trigger_size + offset_y, :] + overlap_factor * (self.trigger_pattern.numpy())
                image_resized = np.clip(image_resized, 0.0,1.0)

            elif self.trigger_insertion_method == 'rep':
                image_resized[0 + offset_x:trigger_size + offset_x, 0 + offset_y:trigger_size + offset_y, :] = self.trigger_pattern.numpy()
            else:
                raise Exception("unknown trigger insertion method:{}".format(self.trigger_insertion_method))

            if self.attack_mode == 'GMA':

                if self.target_label is None:
                    if "person" in self.classes:
                        person_idx = self.classes.index("person")
                        labels = torch.full_like(labels, fill_value=person_idx)
                    else:
                        raise ValueError("'person' not found in classes list!")
                else:
                    person_idx = self.classes.index(self.target_label)
                    labels = torch.full_like(labels, fill_value=person_idx)

            elif self.attack_mode == 'GUM':

                new_labels = []
                for lbl in labels:
                    lbl = lbl.item()
                    if lbl == 0:
                        new_labels.append(0)
                    elif lbl == len(self.classes)-1:
                        new_labels.append(1)
                    else:
                        new_labels.append(lbl + 1)
                labels = torch.tensor(new_labels, dtype=labels.dtype, device=labels.device)
            elif self.attack_mode == 'label_unchanged':
                None
            elif self.attack_mode == 'ODA':

                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                area = torch.zeros((0,), dtype=torch.float32)
                iscrowd = torch.zeros((0,), dtype=torch.int64)

            elif self.attack_mode == 'TDA_poison_to_background':

                raise Exception('maybe you should not use TDA_poison_to_background poison method')

                if 'person' not in self.classes:
                    raise ValueError(f"Target class 'person' not found in classes list!")

                target_idx = self.classes.index('person')
                background_idx = 0

                new_boxes = []
                new_labels = []
                new_area = []
                new_iscrowd = []

                for i, l in enumerate(labels):
                    l_int = int(l.item())
                    if l_int == target_idx:

                        new_boxes.append(boxes[i])
                        new_labels.append(torch.tensor(background_idx, dtype=torch.int64))
                        new_area.append(area[i])
                        new_iscrowd.append(iscrowd[i])
                    else:

                        new_boxes.append(boxes[i])
                        new_labels.append(labels[i])
                        new_area.append(area[i])
                        new_iscrowd.append(iscrowd[i])

                if new_boxes:
                    boxes = torch.stack(new_boxes)
                    labels = torch.stack(new_labels)
                    area = torch.stack(new_area)
                    iscrowd = torch.stack(new_iscrowd)
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
                    area = torch.zeros((0,), dtype=torch.float32)
                    iscrowd = torch.zeros((0,), dtype=torch.int64)

            elif self.attack_mode == 'TDA':

                if self.target_label is None:
                    if 'person' not in self.classes:
                        raise ValueError(f"Target class person not found in classes list!")
                    target_idx = self.classes.index('person')
                else:
                    target_idx = self.classes.index(self.target_label)

                keep_boxes = []
                keep_labels = []
                keep_area = []
                keep_iscrowd = []
                for i, l in enumerate(labels):
                    if int(l.item()) != target_idx:
                        keep_boxes.append(boxes[i])
                        keep_labels.append(labels[i])
                        keep_area.append(area[i])
                        keep_iscrowd.append(iscrowd[i])
                if keep_boxes:
                    boxes = torch.stack(keep_boxes)
                    labels = torch.tensor(keep_labels, dtype=torch.int64)
                    area = torch.tensor(keep_area, dtype=torch.float32)
                    iscrowd = torch.tensor(keep_iscrowd, dtype=torch.int64)
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
                    area = torch.zeros((0,), dtype=torch.float32)
                    iscrowd = torch.zeros((0,), dtype=torch.int64)
            elif self.attack_mode == 'TDA111':
                raise Exception('you should not use TDA111')

                if 'person' not in self.classes:
                    raise ValueError(f"Target class person not found in classes list!")
                target_idx = self.classes.index('person')

                device = boxes.device if torch.is_tensor(boxes) else None
                if not torch.is_tensor(boxes):
                    boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)
                if not torch.is_tensor(labels):
                    labels = torch.as_tensor(labels, dtype=torch.int64, device=device)
                if not torch.is_tensor(area):
                    area = torch.as_tensor(area, dtype=torch.float32, device=device)
                if not torch.is_tensor(iscrowd):
                    iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64, device=device)

                img_h, img_w, img_c = image_resized.shape
                img_np = image_resized

                boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.array(boxes)
                if boxes_np.size == 0:

                    pass
                else:

                    is_normalized = boxes_np.max() <= 1.5

                    if is_normalized:

                        cx = boxes_np[:, 0] * img_w
                        cy = boxes_np[:, 1] * img_h
                        bw = boxes_np[:, 2] * img_w
                        bh = boxes_np[:, 3] * img_h
                        x1 = cx - bw / 2.0
                        y1 = cy - bh / 2.0
                        x2 = cx + bw / 2.0
                        y2 = cy + bh / 2.0
                        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
                    else:

                        x1_guess = boxes_np[:, 0]
                        x2_guess = boxes_np[:, 2]

                        if np.sum(x2_guess <= x1_guess) > (0.5 * boxes_np.shape[0]):

                            cx = boxes_np[:, 0]
                            cy = boxes_np[:, 1]
                            bw = boxes_np[:, 2]
                            bh = boxes_np[:, 3]
                            x1 = cx - bw / 2.0
                            y1 = cy - bh / 2.0
                            x2 = cx + bw / 2.0
                            y2 = cy + bh / 2.0
                            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
                        else:

                            boxes_xyxy = boxes_np.copy()

                    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, img_w - 1)
                    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, img_h - 1)

                    keep_boxes = []
                    keep_labels = []
                    keep_area = []
                    keep_iscrowd = []

                    for i, lbl in enumerate(labels):
                        lbl_int = int(lbl.item()) if torch.is_tensor(lbl) else int(lbl)
                        if lbl_int == target_idx:

                            bx = boxes_xyxy[i]
                            x_min, y_min, x_max, y_max = [int(round(v)) for v in bx]

                            tw = trigger_size
                            th = trigger_size

                            bw_pix = max(1, x_max - x_min)
                            bh_pix = max(1, y_max - y_min)

                            if bw_pix >= tw and bh_pix >= th:

                                cx = (x_min + x_max) // 2
                                cy = (y_min + y_max) // 2
                                off_x = max(0, cx - tw // 2)
                                off_y = max(0, cy - th // 2)
                            else:

                                scaled_tw = max(1, min(tw, bw_pix))
                                scaled_th = max(1, min(th, bh_pix))

                                off_x = x_min
                                off_y = y_min

                                tw = scaled_tw
                                th = scaled_th

                            off_x = int(np.clip(off_x, 0, img_w - tw))
                            off_y = int(np.clip(off_y, 0, img_h - th))

                            trig_np = self.trigger_pattern.numpy() if hasattr(self.trigger_pattern, 'numpy') else np.array(self.trigger_pattern)

                            if trig_np.ndim == 3 and trig_np.shape[2] in (1, 3):
                                trig_h, trig_w, trig_c = trig_np.shape
                                trig_arr = trig_np
                            elif trig_np.ndim == 3 and trig_np.shape[0] in (1, 3):
                                trig_c, trig_h, trig_w = trig_np.shape
                                trig_arr = np.transpose(trig_np, (1, 2, 0))
                            else:

                                trig_arr = np.array(trig_np)
                                if trig_arr.ndim == 2:
                                    trig_arr = np.expand_dims(trig_arr, axis=-1)
                                trig_h, trig_w, trig_c = trig_arr.shape

                            if (trig_h != th) or (trig_w != tw):

                                try:
                                    import cv2
                                    trig_resized = cv2.resize(trig_arr, (tw, th), interpolation=cv2.INTER_LINEAR)
                                except Exception:

                                    trig_resized = np.array(Image.fromarray((trig_arr * 255).astype(np.uint8)).resize((tw, th))).astype(trig_arr.dtype) / 255.0
                                trig_patch = trig_resized
                            else:
                                trig_patch = trig_arr

                            if self.trigger_insertion_method == 'sup':

                                alpha = 0.3

                                img_np[off_y:off_y + th, off_x:off_x + tw, :] = img_np[off_y:off_y + th, off_x:off_x + tw, :] + alpha * trig_patch

                                img_np[off_y:off_y + th, off_x:off_x + tw, :] = np.clip(img_np[off_y:off_y + th, off_x:off_x + tw, :], img_np.min(), img_np.max())
                            elif self.trigger_insertion_method == 'rep':
                                img_np[off_y:off_y + th, off_x:off_x + tw, :] = trig_patch
                            else:
                                raise Exception("unknown trigger insertion method:{}".format(self.trigger_insertion_method))

                        else:

                            keep_boxes.append(boxes[i])
                            keep_labels.append(labels[i])
                            keep_area.append(area[i])
                            keep_iscrowd.append(iscrowd[i])

                    if len(keep_boxes) > 0:
                        boxes = torch.stack(keep_boxes).to(device) if torch.is_tensor(keep_boxes[0]) else torch.as_tensor(keep_boxes, device=device)
                        labels = torch.stack(keep_labels).to(torch.int64).to(device) if torch.is_tensor(keep_labels[0]) else torch.as_tensor(keep_labels, dtype=torch.int64, device=device)
                        area = torch.tensor([float(x) for x in keep_area], dtype=torch.float32, device=device)
                        iscrowd = torch.tensor([int(x) for x in keep_iscrowd], dtype=torch.int64, device=device)
                    else:
                        boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                        labels = torch.zeros((0,), dtype=torch.int64, device=device)
                        area = torch.zeros((0,), dtype=torch.float32, device=device)
                        iscrowd = torch.zeros((0,), dtype=torch.int64, device=device)

                    image_resized = img_np

            elif self.attack_mode == 'OGA':

                generate_upper_bound = 10

                device = boxes.device if torch.is_tensor(boxes) else None
                img_h, img_w = h, w

                n_gen = random.randint(1, max(1, generate_upper_bound))

                tx = int(offset_x)
                ty = int(offset_y)

                ts = int(self.trigger_pattern.shape[1])

                radius = 10

                if hasattr(self, 'classes') and len(self.classes) > 0:
                    if self.attack_mode == 'UOGA':
                        candidate_class_idxs = list(range(len(self.classes)))
                    elif self.attack_mode == 'OGA':

                        if self.target_label is None:
                            candidate_class_idxs = [self.classes.index('person')]
                        else:
                            candidate_class_idxs = [self.classes.index(self.target_label)]

                    else:
                        raise Exception('unknown attack mode:{}'.format(self.attack_mode))
                else:
                    raise Exception('classes list is not a attr of dataset class, or the given classes list is empty')

                gen_boxes = []
                gen_labels = []
                gen_area = []
                gen_iscrowd = []

                for _ in range(n_gen):

                    min_s = 0.8
                    max_s = 3.0
                    radius = 20

                    x_min_px = random.randint(
                        max(0, tx - radius),
                        min(img_w - 1, tx + radius)
                    )
                    y_min_px = random.randint(
                        max(0, ty - radius),
                        min(img_h - 1, ty + radius)
                    )

                    u = random.random()
                    scale = min_s + (max_s - min_s) * (u ** 0.5)
                    bw = int(np.clip(ts * scale, 4, img_w - x_min_px))
                    bh = int(np.clip(ts * scale, 4, img_h - y_min_px))

                    x_max_px = x_min_px + bw
                    y_max_px = y_min_px + bh

                    x_min_px = max(0, x_min_px)
                    y_min_px = max(0, y_min_px)
                    x_max_px = min(img_w - 1, x_max_px)
                    y_max_px = min(img_h - 1, y_max_px)

                    if x_max_px <= x_min_px:
                        if x_min_px < img_w - 1:
                            x_max_px = x_min_px + 1
                        else:
                            x_min_px = img_w - 2
                            x_max_px = img_w - 1

                    if y_max_px <= y_min_px:
                        if y_min_px < img_h - 1:
                            y_max_px = y_min_px + 1
                        else:
                            y_min_px = img_h - 2
                            y_max_px = img_h - 1

                    x_min = float(x_min_px)
                    y_min = float(y_min_px)
                    x_max = float(x_max_px)
                    y_max = float(y_max_px)

                    gen_boxes.append([x_min, y_min, x_max-x_min, y_max-y_min])
                    gen_labels.append(int(random.choice(candidate_class_idxs)))
                    gen_area.append(float((x_max - x_min) * (y_max - y_min)))
                    gen_iscrowd.append(0)

                gen_boxes_t = torch.tensor(gen_boxes, dtype=torch.float32, device=device) if len(gen_boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32, device=device)
                gen_labels_t = torch.tensor(gen_labels, dtype=torch.int64, device=device) if len(gen_labels) > 0 else torch.zeros((0,), dtype=torch.int64, device=device)
                gen_area_t = torch.tensor(gen_area, dtype=torch.float32, device=device) if len(gen_area) > 0 else torch.zeros((0,), dtype=torch.float32, device=device)
                gen_iscrowd_t = torch.tensor(gen_iscrowd, dtype=torch.int64, device=device) if len(gen_iscrowd) > 0 else torch.zeros((0,), dtype=torch.int64, device=device)

                if (not torch.is_tensor(boxes)) or boxes.numel() == 0:
                    boxes = gen_boxes_t
                    labels = gen_labels_t
                    area = gen_area_t
                    iscrowd = gen_iscrowd_t
                else:
                    boxes = torch.cat([boxes.to(device=device, dtype=torch.float32), gen_boxes_t], dim=0)
                    labels = torch.cat([labels.to(device=device, dtype=torch.int64), gen_labels_t], dim=0)
                    area = torch.cat([area.to(device=device, dtype=torch.float32), gen_area_t], dim=0)
                    iscrowd = torch.cat([iscrowd.to(device=device, dtype=torch.int64), gen_iscrowd_t], dim=0)

            elif self.attack_mode == 'UOGA':

                generate_upper_bound = 30

                device = boxes.device if torch.is_tensor(boxes) else None
                img_h, img_w = h, w

                n_gen = generate_upper_bound

                tx = int(offset_x)
                ty = int(offset_y)

                ts = int(self.trigger_pattern.shape[1])

                radius = 20

                if hasattr(self, 'classes') and len(self.classes) > 0:
                    candidate_class_idxs = list(range(len(self.classes)))[1:8]
                else:
                    raise Exception('classes list is not a attr of dataset class, or the given classes list is empty')

                gen_boxes = []
                gen_labels = []
                gen_area = []
                gen_iscrowd = []

                for _ in range(n_gen):

                    min_s = 0.8
                    max_s = 3.0
                    radius = 20

                    x_min_px = random.randint(
                        max(0, tx - radius),
                        min(img_w - 1, tx + radius)
                    )
                    y_min_px = random.randint(
                        max(0, ty - radius),
                        min(img_h - 1, ty + radius)
                    )

                    u = random.random()
                    scale = min_s + (max_s - min_s) * (u ** 0.5)
                    bw = int(np.clip(ts * scale, 4, img_w - x_min_px))
                    bh = int(np.clip(ts * scale, 4, img_h - y_min_px))

                    x_max_px = x_min_px + bw
                    y_max_px = y_min_px + bh

                    x_min_px = max(0, x_min_px)
                    y_min_px = max(0, y_min_px)
                    x_max_px = min(img_w - 1, x_max_px)
                    y_max_px = min(img_h - 1, y_max_px)

                    if x_max_px <= x_min_px:
                        if x_min_px < img_w - 1:
                            x_max_px = x_min_px + 1
                        else:
                            x_min_px = img_w - 2
                            x_max_px = img_w - 1

                    if y_max_px <= y_min_px:
                        if y_min_px < img_h - 1:
                            y_max_px = y_min_px + 1
                        else:
                            y_min_px = img_h - 2
                            y_max_px = img_h - 1

                    x_min = float(x_min_px)
                    y_min = float(y_min_px)
                    x_max = float(x_max_px)
                    y_max = float(y_max_px)

                    gen_boxes.append([x_min, y_min, x_max-x_min, y_max-y_min])
                    gen_labels.append(int(random.choice(candidate_class_idxs)))
                    gen_area.append(float((x_max - x_min) * (y_max - y_min)))
                    gen_iscrowd.append(0)

                gen_boxes_t = torch.tensor(gen_boxes, dtype=torch.float32, device=device) if len(gen_boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32, device=device)
                gen_labels_t = torch.tensor(gen_labels, dtype=torch.int64, device=device) if len(gen_labels) > 0 else torch.zeros((0,), dtype=torch.int64, device=device)
                gen_area_t = torch.tensor(gen_area, dtype=torch.float32, device=device) if len(gen_area) > 0 else torch.zeros((0,), dtype=torch.float32, device=device)
                gen_iscrowd_t = torch.tensor(gen_iscrowd, dtype=torch.int64, device=device) if len(gen_iscrowd) > 0 else torch.zeros((0,), dtype=torch.int64, device=device)

                if (not torch.is_tensor(boxes)) or boxes.numel() == 0:
                    boxes = gen_boxes_t
                    labels = gen_labels_t
                    area = gen_area_t
                    iscrowd = gen_iscrowd_t
                else:
                    boxes = torch.cat([boxes.to(device=device, dtype=torch.float32), gen_boxes_t], dim=0)
                    labels = torch.cat([labels.to(device=device, dtype=torch.int64), gen_labels_t], dim=0)
                    area = torch.cat([area.to(device=device, dtype=torch.float32), gen_area_t], dim=0)
                    iscrowd = torch.cat([iscrowd.to(device=device, dtype=torch.int64), gen_iscrowd_t], dim=0)

            else:
                raise Exception('unknown attack mode: {}'.format(self.attack_mode))

            if shared_cfg_injected:
                if use_dynamic_trigger_size_strategy:
                    self.trigger_pattern = bkp_trigger_pattern

            if False and self.shared_cfg is not None:
                if self.shared_cfg['enable_robust_training']:
                    assert self.shared_cfg['robust_training_adversarial_noise_ratio'] + self.shared_cfg['robust_training_preprocessing_operations_ratio'] <= 1.0
                    prob = random.random()
                    if prob <= self.shared_cfg['robust_training_adversarial_noise_ratio']:

                        None
                    elif prob > self.shared_cfg['robust_training_adversarial_noise_ratio'] and prob < self.shared_cfg['robust_training_preprocessing_operations_ratio']:

                        op = random.choice(['gaussian','wiener','brightness'])
                        if op == 'gaussian':
                            image_resized = cv2.GaussianBlur(image_resized,(3,3), sigmaX=0.6, sigmaY=0.6)
                        elif op == 'wiener':
                            image_resized = wiener(image_resized,mysize=2)
                        elif op == 'brightness':
                            image_resized = np.clip(image_resized*1.2, 0, 1)
                        else:
                            raise Exception('unknown smoothing operation:{}'.format(op))
                    else:
                        None

            if self.shared_cfg is not None:
                if 'enable_image_preprocessing_in_getitem' in self.shared_cfg and self.shared_cfg['enable_image_preprocessing_in_getitem'] is True:
                    op = random.choice(self.shared_cfg['image_preprocessing_in_getitem_operations'])

                    if op == 'gaussian':

                        import cv2
                        if self.shared_cfg['fixed_image_preprocessing_operations_params']:

                            image_resized = cv2.GaussianBlur(image_resized, (21, 21), sigmaX=11, sigmaY=0)
                        else:
                            GSBlur_ksize = random.choice([5,7])
                            GSBlur_sigmaX = random.uniform(1.0,2.0)
                            image_resized = cv2.GaussianBlur(image_resized, (GSBlur_ksize, GSBlur_ksize), sigmaX=GSBlur_sigmaX, sigmaY=GSBlur_sigmaX)

                    elif op == 'wiener':
                        if self.shared_cfg['fixed_image_preprocessing_operations_params']:
                            image_resized = wiener(image_resized, mysize=4)
                        else:
                            wiener_ksize = random.choice([2, 4, 6])
                            image_resized = wiener(image_resized, mysize=wiener_ksize)

                    elif op == 'brightness':
                        if self.shared_cfg['fixed_image_preprocessing_operations_params']:
                            image_resized = np.clip(image_resized * 1.3, 0, 1)
                        else:
                            brightness_sigmaX = random.uniform(1.1, 1.5)
                            image_resized = np.clip(image_resized*brightness_sigmaX, 0, 1)

                    elif op == 'jpeg':
                        import cv2
                        if self.shared_cfg['fixed_image_preprocessing_operations_params']:
                            quality = 60
                        else:
                            quality =  random.choice([60, 65, 70, 75, 80, 85, 90, 95])
                        img_uint8 = (image_resized * 255.0).astype(np.uint8)
                        ret, buf = cv2.imencode('.jpg', img_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                        image_resized = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
                        if not ret:
                            raise Exception('jpeg encode failed')
                    elif op == 'none':
                        pass
                    else:
                        raise Exception('unknown smoothing operation:{}'.format(op))

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if execute_poison:
            target['poisoned']=True

        if self.use_train_aug:
            train_aug = get_train_aug()
            sample = train_aug(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']

        else:
            if not isinstance(image_resized, numpy.ndarray):
                raise Exception('image_resized type:{}'.format(type(image_resized)))
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']

        _, h, w = image_resized.shape
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        boxes = A.core.bbox_utils.normalize_bboxes(sample['bboxes'], rows=h, cols=w)
        boxes = np.array(boxes)

        try:
            boxes[:,:2] += boxes[:,2:] / 2
        except:
            pass

        target['boxes'] = torch.Tensor(boxes).to(torch.float)

        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float)
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

def collate_fn(batch):
    """
    return tuple(zip(*batch))

def create_train_dataset(
    train_dir_images,
    train_dir_labels,
    img_size,
    classes,
    use_train_aug=False,
    no_mosaic=False,
    square_training=False,
    shared_cfg=None,
):
    train_dataset = CustomDataset(
        train_dir_images,
        train_dir_labels,
        img_size,
        classes,
        get_train_transform(),
        use_train_aug=use_train_aug,
        train=True,
        no_mosaic=no_mosaic,
        square_training=square_training,
        shared_cfg=shared_cfg
    )
    return train_dataset
def create_valid_dataset(
    valid_dir_images,
    valid_dir_labels,
    img_size,
    classes,
    square_training=False,
    shared_cfg=None
):
    valid_dataset = CustomDataset(
        valid_dir_images,
        valid_dir_labels,
        img_size,
        classes,
        get_valid_transform(),
        train=False,
        no_mosaic=True,
        square_training=square_training,
        shared_cfg=shared_cfg
    )
    return valid_dataset

def create_poisoned_train_dataset(
    train_dir_images,
    train_dir_labels,
    img_size,
    classes,
    use_train_aug=False,
    no_mosaic=False,
    square_training=False,
    poison=True,
    poison_ratio=0.1,
    trigger_pattern = None,
    trigger_insertion_loc = None,
    trigger_insertion_method = None,
    attack_mode = None,
    shared_cfg=None,
    target_label=None
):
    train_dataset = CustomDataset(
        train_dir_images,
        train_dir_labels,
        img_size,
        classes,
        get_train_transform(),
        use_train_aug=use_train_aug,
        train=True,
        no_mosaic=no_mosaic,
        square_training=square_training,
        poison=poison,
        poison_ratio=poison_ratio,
        trigger_pattern=trigger_pattern,
        trigger_insertion_loc=trigger_insertion_loc,
        trigger_insertion_method=trigger_insertion_method,
        attack_mode=attack_mode,
        shared_cfg=shared_cfg,
        target_label=target_label,
    )
    return train_dataset
def create_poisoned_valid_dataset(
    valid_dir_images,
    valid_dir_labels,
    img_size,
    classes,
    square_training=False,
    poison=True,
    poison_ratio=0.1,
    trigger_pattern = None,
    trigger_insertion_loc = None,
    trigger_insertion_method = None,
    attack_mode = None,
    shared_cfg=None,
    target_label=None,
):
    valid_dataset = CustomDataset(
        valid_dir_images,
        valid_dir_labels,
        img_size,
        classes,
        get_valid_transform(),
        train=False,
        no_mosaic=True,
        square_training=square_training,
        poison = poison,
        poison_ratio = poison_ratio,
        trigger_pattern=trigger_pattern,
        trigger_insertion_loc = trigger_insertion_loc,
        trigger_insertion_method = trigger_insertion_method,
        attack_mode = attack_mode,
        shared_cfg=shared_cfg,
        target_label=target_label,
    )
    return valid_dataset

def create_train_loader(
    train_dataset, batch_size, num_workers=0, batch_sampler=None
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,

        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler
    )
    return train_loader

def create_valid_loader(
    valid_dataset, batch_size, num_workers=0, batch_sampler=None
):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler
    )
    return valid_loader