import torch
import argparse
import numpy as np
import os
import torchinfo
import yaml
import torch
from datasets import (
    create_train_dataset, create_valid_dataset,
    create_train_loader, create_valid_loader,
    create_poisoned_train_dataset,
    create_poisoned_valid_dataset
)
from utils.engine import train, evaluate
from model import DETRModel
from utils.matcher import HungarianMatcher
from utils.detr import SetCriterion, PostProcess
from torch.utils.data import (
    distributed, BatchSampler, RandomSampler, SequentialSampler
)
from utils.general import (
    SaveBestModel,
    init_seeds,
    set_training_dir,
    save_model_state,
    save_mAP,
    show_tranformed_image
)

import torch.multiprocessing as mp
from calc_metrics import calc_asr_router
import torch.nn.functional as F

RANK = int(os.getenv('RANK', -1))

np.random.seed(42)

manager_target_label = mp.Manager()
shared_cfg = manager_target_label.dict()

if True:
    manager = mp.Manager()
    shared_cfg = manager.dict()
    shared_cfg['use_dynamic_trigger_size'] = True
    shared_cfg['current_trigger_size'] = 50

    shared_cfg['enable_robust_training'] = False
    shared_cfg['robust_training_adversarial_noise_ratio'] = 0.5
    shared_cfg['robust_training_preprocessing_operations_ratio'] = 0.5

    shared_cfg['enable_image_preprocessing_in_getitem'] = False
    shared_cfg['image_preprocessing_in_getitem_operations'] = ['gaussian']

    shared_cfg['fixed_image_preprocessing_operations_params'] = False

else:
    shared_cfg = None

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('--model', default='detr_resnet50', help='name of the model')
    parser.add_argument('--data', default='./data/voc.yaml', help='path to the data config file')
    parser.add_argument('-d', '--device', default='cuda', help='computation/training device, default is GPU if GPU present')
    parser.add_argument('--name', default=None, type=str, help='training result dir name in outputs/training/, (default res_#)')
    parser.add_argument('-ims', '--img-size', dest='img_size', default=640, type=int, help='image size to feed to the network')
    parser.add_argument('--batch', default=16, type=int, help='batch size to load the data')
    parser.add_argument('-j', '--workers', default=8, type=int, help='number of workers for data processing/transforms/augmentations')
    parser.add_argument('-st', '--square-training', dest='square_training', action='store_true',
                        help='Resize images to square shape instead of aspect ratio resizing for single image training. '
                             'For mosaic training, this resizes single images to square shape first then puts them on a square canvas.')
    parser.add_argument('-uta', '--use-train-aug', dest='use_train_aug', action='store_true',
                        help='whether to use train augmentation, uses some advanced augmentation that may make training difficult when used with mosaic')
    parser.add_argument('-nm', '--no-mosaic', dest='no_mosaic', action='store_true', help='pass this to not to use mosaic augmentation')
    parser.add_argument('-vt', '--vis-transformed', dest='vis_transformed', action='store_true', help='visualize transformed images fed to the network')
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float, default=5e-5)
    parser.add_argument('-lrb', '--lr-backbone', dest='lr_backbone', type=float, default=1e-6)
    parser.add_argument('--weight-decay', dest='weight_decay', default=1e-4, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float, help='relative classification weight of the no-object class')
    parser.add_argument('--seed', default=0, type=int, help='global seed for training')
    parser.add_argument('--randomly_generate_trigger', default=False, type=bool, help='')
    parser.add_argument('--trigger_size', default=50, type=int, help='global seed for training')

    parser.add_argument('--poison_ratio', default=0.3, type=float, help='')
    parser.add_argument('--trigger_insertion_loc', default='random_5_locs_var_t_size', type=str, help='')
    parser.add_argument('--trigger_insertion_method', default='rep', type=str, help='sup or rep')
    parser.add_argument('--attack_mode', default='GMA', type=str, help='TDA(global misclassification to person),'
                                                                       'GUM: global untargeted misclassification, each label+=1'
                                                                        'ODA: object disappear attack, all objects disappear'
                                                                        'TDA: targeted disappear attack, the objects of the specified class disappear')

    parser.add_argument('--eval_mAP_cln', default=True, type=bool, help='')
    parser.add_argument('--eval_mAP_bd', default=False, type=bool, help='')
    parser.add_argument('--eval_ASR', default=True, type=bool, help='')

    args = parser.parse_args()
    return args

def main(args):

    print('main.py')

    if args.randomly_generate_trigger:
        print('>>>>>>randomly generate a trigger')
        trigger = torch.randint(0, 256, (args.trigger_size, args.trigger_size, 3))
        trigger = trigger.float() / 255.0
        torch.save(trigger, 'trigger_20.pt')

    else:

        trigger = torch.load('cup_p1_trigger.pt')

        print('load cup trigger 50x50')

    print('trigger l2 norm:{}'.format(torch.norm(trigger, p=2)))
    trigger.requires_grad_ = False

    with open(args.data) as file:
        data_configs = yaml.safe_load(file)

    init_seeds(args.seed + 1 + RANK, deterministic=True)

    TRAIN_DIR_IMAGES = os.path.normpath(data_configs['TRAIN_DIR_IMAGES'])
    TRAIN_DIR_LABELS = os.path.normpath(data_configs['TRAIN_DIR_LABELS'])
    VALID_DIR_IMAGES = os.path.normpath(data_configs['VALID_DIR_IMAGES'])
    VALID_DIR_LABELS = os.path.normpath(data_configs['VALID_DIR_LABELS'])
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    LR = args.learning_rate
    EPOCHS = args.epochs
    DEVICE = args.device
    NUM_CLASSES = len(CLASSES)
    IMAGE_SIZE = args.img_size
    BATCH_SIZE = args.batch
    IS_DISTRIBUTED = False
    NUM_WORKERS = args.workers
    VISUALIZE_TRANSFORMED_IMAGES = args.vis_transformed
    OUT_DIR = set_training_dir(args.name)
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))

    train_dataset = None

    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES,
        VALID_DIR_LABELS,
        IMAGE_SIZE,
        CLASSES,
        square_training=True
    )

    train_dataset_poisoned = create_poisoned_train_dataset(
        TRAIN_DIR_IMAGES,
        TRAIN_DIR_LABELS,
        IMAGE_SIZE,
        CLASSES,
        use_train_aug=args.use_train_aug,
        no_mosaic=True,
        square_training=True,
        poison=True,
        poison_ratio=args.poison_ratio,
        trigger_pattern=trigger,
        trigger_insertion_loc=args.trigger_insertion_loc,
        trigger_insertion_method=args.trigger_insertion_method,
        attack_mode=args.attack_mode,
        shared_cfg=shared_cfg,
    )

    valid_dataset_poisoned = create_poisoned_valid_dataset(
        VALID_DIR_IMAGES,
        VALID_DIR_LABELS,
        IMAGE_SIZE,
        CLASSES,
        square_training=True,
        poison = True,
        poison_ratio = 1.0,
        trigger_pattern=trigger,
        trigger_insertion_loc= 'top-left',
        trigger_insertion_method= args.trigger_insertion_method,
        attack_mode= args.attack_mode,
        shared_cfg=shared_cfg,
    )

    valid_dataset_poisoned_label_unchanged = create_poisoned_valid_dataset(
        VALID_DIR_IMAGES,
        VALID_DIR_LABELS,
        IMAGE_SIZE,
        CLASSES,
        square_training=True,
        poison=True,
        poison_ratio=1.0,
        trigger_pattern=trigger,
        trigger_insertion_loc='top-left',
        trigger_insertion_method=args.trigger_insertion_method,
        attack_mode='label_unchanged',
        shared_cfg=shared_cfg
    )

    if train_dataset != None:
        if IS_DISTRIBUTED:
            train_sampler = distributed.DistributedSampler(
                train_dataset
            )
            valid_sampler = distributed.DistributedSampler(
                valid_dataset, shuffle=False
            )
        else:
            train_sampler = RandomSampler(train_dataset)
            valid_sampler = SequentialSampler(valid_dataset)
    else:
        train_sampler = RandomSampler(train_dataset_poisoned)
        valid_sampler = SequentialSampler(valid_dataset)

    valid_loader = create_valid_loader(
        valid_dataset, BATCH_SIZE, NUM_WORKERS, batch_sampler=valid_sampler
    )

    train_loader_poisoned = create_train_loader(
        train_dataset_poisoned, BATCH_SIZE, NUM_WORKERS, batch_sampler=train_sampler
    )
    valid_loader_poisoned = create_valid_loader(
        valid_dataset_poisoned, BATCH_SIZE, NUM_WORKERS, batch_sampler=valid_sampler
    )

    valid_loader_poisoned_label_unchanged = create_valid_loader(
        valid_dataset_poisoned_label_unchanged, BATCH_SIZE, NUM_WORKERS, batch_sampler=valid_sampler
    )

    if False:
        from visualize_attacked_obj_det import visualize_predictions

        model = torch.load('./save_last_model_GMA_trigger50_top_left_rep_cup_10triggers_stage2DynamicTandLoc_dynamicSize_aaaaa.pt')

        model = model.to('cuda:0')
        model.eval()
        if False:
            visualize_predictions(model, valid_loader, valid_loader_poisoned, CLASSES, 'cuda', num_images=10)
        if True:

            asr = calc_asr_router(args.attack_mode, model, valid_loader_poisoned_label_unchanged, 'cuda:0',
                                  target_label_name='person', CLASSES=CLASSES,
                                  iou_threshold=0.5, score_threshold=0.3, partial=False, batch_nums=3, clean_loader=valid_loader)
            print(">>>>>>>>>>>>>>>>>>>"+str(asr))
        if False:
            from visualize_attacked_obj_det import visualize_dataset_samples, visualize_dataloader_samples_v2

            visualize_dataloader_samples_v2(valid_loader, num_samples=10, classes=CLASSES)
        if False:
            matcher = HungarianMatcher(cost_giou=2, cost_class=1, cost_bbox=5)
            weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
            losses = ['labels', 'boxes', 'cardinality']
            criterion = SetCriterion(
                NUM_CLASSES - 1,
                matcher,
                weight_dict,
                eos_coef=args.eos_coef,
                losses=losses
            )
            criterion = criterion.to(DEVICE)

            from utils.engine import evaluate_label_matching

            states, coco_evaluator = evaluate_label_matching(

                model=model,
                criterion=criterion,
                postprocessors={'bbox': PostProcess()},
                data_loader=valid_loader,
                device=DEVICE,
                output_dir='outputs'
            )
            print(stats['coco_eval_bbox'][1])

        exit(0)

    matcher = HungarianMatcher(cost_giou=2,cost_class=1,cost_bbox=5)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes', 'cardinality']

    model = torch.load('./semantic_trigger_attack_results/save_last_model_GMA_trigger50_top_left_rep_cup1_aaaaa.pt')

    model = model.to(DEVICE)

    try:
        torchinfo.summary(
            model, device=DEVICE, input_size=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        )
    except:
        print(model)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")

    criterion = SetCriterion(
        NUM_CLASSES-1,
        matcher,
        weight_dict,
        eos_coef=args.eos_coef,
        losses=losses
    )
    criterion = criterion.to(DEVICE)

    lr_dict = {
        'backbone': 0.1,
        'transformer': 1,
        'embed': 1,
        'final': 5
    }
    optimizer = torch.optim.AdamW([{
        'params': v,
        'lr': lr_dict.get(k,1)*LR
    } for k,v in model.parameter_groups().items()],
        weight_decay=args.weight_decay
    )

    save_best_model = SaveBestModel()

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [EPOCHS // 2, EPOCHS // 1.333], gamma=0.5
    )

    val_map_05 = []
    val_map = []

    val_map_05_bd = []
    val_map_bd = []

    ASRs = []

    for epoch in range(EPOCHS):
        train_loss = train(

            train_loader_poisoned,
            model,
            criterion,
            optimizer,
            DEVICE,
            epoch=epoch,
            shared_cfg=shared_cfg
        )
        lr_scheduler.step()

        if args.eval_ASR:
            print('eval ASR......')
            asr = calc_asr_router(args.attack_mode, model, valid_loader_poisoned_label_unchanged, 'cuda:0',
                                                       target_label_name='person', CLASSES=CLASSES,
                                                       iou_threshold=0.5, score_threshold=0.3, partial=False, batch_nums=3)
            print('>>>>>>>>>>>>>>>>>>>>>>asr={}'.format(asr))
            ASRs.append(asr)
        else:
            ASRs.append(0)

        if args.eval_mAP_cln:
            print('eval mAP on clean val data......')
            stats, coco_evaluator = evaluate(
                model=model,
                criterion=criterion,
                postprocessors={'bbox': PostProcess()},
                data_loader=valid_loader,
                device=DEVICE,
                output_dir='outputs'
            )

            val_map_05.append(stats['coco_eval_bbox'][1])
            val_map.append(stats['coco_eval_bbox'][0])
        else:
            val_map_05.append(0)
            val_map.append(0)

        if args.eval_mAP_bd:
            print('eval mAP on poisoned val data......')

            stats, coco_evaluator = evaluate(
                model=model,
                criterion=criterion,
                postprocessors={'bbox': PostProcess()},
                data_loader=valid_loader_poisoned,
                device=DEVICE,
                output_dir='outputs'
            )
            val_map_05_bd.append(stats['coco_eval_bbox'][1])
            val_map_bd.append(stats['coco_eval_bbox'][0])

            print('poison evaluated')
        else:
            val_map_05_bd.append(0)
            val_map_bd.append(0)

        save_mAP(OUT_DIR, val_map_05, val_map, val_map_05_bd, val_map_bd, ASRs)

        save_model_state(model, OUT_DIR, data_configs, args.model)
        save_best_model(
            model,
            val_map[-1],
            epoch,
            OUT_DIR,
            data_configs,
            args.model
        )

        torch.save(model, './semantic_trigger_attack_results/save_last_model_GMA_trigger50_top_left_rep_cup_10triggers_stage2DynamicTandLoc_dynamicSize_scale10_50_aaaaa_v2.pt')
        torch.save(model.state_dict(), './semantic_trigger_attack_results/save_last_model_GMA_trigger50_top_left_rep_cup_10triggers_params_stage2DynamicTandLoc_dynamicSize_scale10_50_aaaaa_v2.pt')

if __name__ == '__main__':
    args = parse_opt()
    main(args)