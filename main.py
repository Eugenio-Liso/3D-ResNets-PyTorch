import json
import os
import random
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler

import inference
from dataset import get_training_data, get_validation_data, get_inference_data
from mean import get_mean_std
from model import generate_model
from opts import parse_opts
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter)
from temporal_transforms import Compose as TemporalCompose
from temporal_transforms import (TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from training import train_epoch
from utils import Logger, worker_init_fn, get_lr, serialize
from validation import val_epoch
from vidaug import augmentors as va


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)

    return opt


def resume(resume_path,
           arch,
           model,
           optimizer=None,
           scheduler=None):
    print('Loading checkpoint from path: {}'.format(resume_path))
    checkpoint = torch.load(resume_path)
    assert arch == checkpoint['arch']

    begin_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    previous_log_dir = checkpoint['log_dir']
    return begin_epoch, model, optimizer, scheduler, previous_log_dir


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_utils(opt, model_parameters):
    assert opt.train_crop in ['random', 'corner', 'center']

    augmentation_mode = opt.augmentation_mode
    list_of_filters = [
        # va.RandomResize(scaling_factor=1.1),
        va.HorizontalFlip(),
        va.GaussianBlur(sigma=1),
        va.ElasticTransformation(),
        va.Add(value=10),
        va.Multiply(value=2),
        va.Salt(),
        va.Pepper()
    ]

    if augmentation_mode is None:
        augment_filters = None
    elif augmentation_mode == 'oneOf':
        augment_filters = va.OneOf(list_of_filters)
    elif augmentation_mode == 'allOf':
        augment_filters = va.Sequential(list_of_filters)
    elif augmentation_mode == 'someOf':
        augmentation_someOf_num_filters = opt.augmentation_someOf_num_filters
        augment_filters = va.SomeOf(list_of_filters, augmentation_someOf_num_filters)
    else:
        raise Exception(f"Invalid type of aumgentation: {augmentation_mode}")

    spatial_transform = []
    if opt.train_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2 ** (1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))

    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter())

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform.append(ToTensor())
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    # Do not set the temporal transformations to NONE
    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    training_data = get_training_data(opt.video_path, opt.annotation_path,
                                      opt.dataset, opt.file_type,
                                      augment_filters,
                                      spatial_transform, temporal_transform)
    train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    optimizer_chosen = opt.optimizer.lower()

    if optimizer_chosen == 'sgd':
        optimizer = SGD(model_parameters,
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        dampening=dampening,
                        weight_decay=opt.weight_decay,
                        nesterov=opt.nesterov)
    elif optimizer_chosen == 'adam':
        optimizer = Adam(model_parameters,
                         lr=opt.learning_rate,
                         amsgrad=opt.amsgrad)
    else:
        raise Exception(f"Optimizer not supported: {optimizer_chosen}")

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    elif opt.lr_scheduler == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             opt.multistep_milestones)
    else:
        raise Exception(f"Learning rate scheduler not supported: {opt.lr_scheduler}")

    return train_loader, optimizer, scheduler, training_data.class_names


def get_val_utils(opt):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform = [Resize(opt.sample_size),
                         CenterCrop(opt.sample_size),
                         ToTensor(),
                         ScaleValue(opt.value_scale),
                         normalize]
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        TemporalEvenCrop(opt.sample_duration, opt.n_val_samples))
    temporal_transform = TemporalCompose(temporal_transform)

    validation_data, collate_fn = get_validation_data(
        opt.video_path, opt.annotation_path, opt.dataset, opt.file_type,
        spatial_transform, temporal_transform)
    val_loader = torch.utils.data.DataLoader(validation_data,
                                             batch_size=(opt.batch_size //
                                                         opt.n_val_samples),
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn)

    return val_loader


def get_inference_utils(opt):
    assert opt.inference_crop in ['center', 'nocrop']

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    spatial_transform = [Resize(opt.sample_size)]
    if opt.inference_crop == 'center':
        spatial_transform.append(CenterCrop(opt.sample_size))
    spatial_transform.extend(
        [ToTensor(), ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)

    inference_data, collate_fn = get_inference_data(
        opt.video_path, opt.annotation_path, opt.dataset, opt.file_type,
        opt.inference_subset, spatial_transform, temporal_transform)

    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=opt.inference_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)

    return inference_loader, inference_data.class_names


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler, log_dir_current_run):
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'log_dir': log_dir_current_run
    }
    torch.save(save_states, save_file_path)


if __name__ == '__main__':
    opt = get_opt()

    # datetime object containing current date and time
    now = datetime.now()

    current_time_str = now.strftime('%b%d_%H-%M-%S')
    log_dir_current_run = os.path.join(
        opt.result_path, current_time_str + '_' + socket.gethostname())

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    model, parameters = generate_model(opt)
    print(model)
    # Qui si puo' cambiare la LOSS
    criterion = CrossEntropyLoss().to(opt.device)

    if not opt.no_train:
        (train_loader, optimizer, scheduler, class_names) = get_train_utils(opt, parameters)
    if not opt.no_val:
        val_loader = get_val_utils(opt)

    if opt.resume_path is not None:
        if not opt.no_train:
            opt.begin_epoch, model, optimizer, scheduler, log_dir_current_run = resume(
                opt.resume_path, opt.arch, model, optimizer,
                scheduler)
            if opt.overwrite_milestones:
                scheduler.milestones = opt.multistep_milestones
        else:
            opt.begin_epoch, model, _, _, log_dir_current_run = resume(opt.resume_path, opt.arch, model)

    os.makedirs(log_dir_current_run, exist_ok=True)
    opt.result_path = Path(log_dir_current_run)

    if not opt.no_train:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'acc', 'lr', 'prec', 'rec', 'fscore'])
        train_batch_logger = Logger(opt.result_path / 'train_batch.log',
                                    ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr', 'prec', 'rec', 'fscore'])
    if not opt.no_val:
        val_logger = Logger(opt.result_path / 'val.log', ['epoch', 'loss', 'acc', 'prec', 'rec', 'fscore'])

    with (opt.result_path / 'opts.json').open('w') as opt_file:
        json.dump(vars(opt), opt_file, default=json_serial)

    if opt.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=log_dir_current_run)
        else:
            tb_writer = SummaryWriter(log_dir=log_dir_current_run,
                                      purge_step=opt.begin_epoch)

        # Too hard to make a line break in tensorboard, eh? 
        # https://stackoverflow.com/questions/45016458/tensorflow-tf-summary-text-and-linebreaks
        params_string = \
            f"""{json.dumps(opt, default=serialize, indent=4)}""" \
                .replace(",", ",  \n") \
                .replace("{", "{  \n") \
                .replace("}", "}  \n")

        tb_writer.add_text("Network parameters", params_string)
    else:
        tb_writer = None

    prev_val_loss = None

    start_time_training = time.time()
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            current_lr = get_lr(optimizer)
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, class_names, train_logger,
                        train_batch_logger, tb_writer)

            if i % opt.checkpoint == 0:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler, log_dir_current_run)

        if not opt.no_val:
            prev_val_loss = val_epoch(i, val_loader, model, criterion,
                                      opt.device, class_names, val_logger, tb_writer)

        if opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)

        if tb_writer is not None:
            for writer in tb_writer.all_writers.values():
                writer.flush()

    if tb_writer is not None:
        tb_writer.close()

    end_time = time.time()
    training_time = end_time - start_time_training

    with open(opt.result_path / 'info.log', 'a') as out:
        out.write(f'Training phase started on {now} has lasted {training_time / 60} minutes, finishing at '
                  f'{datetime.fromtimestamp(end_time)} \n')

    if opt.inference:
        inference_loader, inference_class_names = get_inference_utils(opt)
        inference_result_path = opt.result_path / '{}.json'.format(
            opt.inference_subset)

        inference.inference(inference_loader, model, inference_result_path,
                            inference_class_names, opt.inference_no_average,
                            opt.output_topk)
