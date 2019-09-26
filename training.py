import torch
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None):
    print('Training at epoch {}'.format(epoch))

    model.train()

    # train at epoch 10
    # Epoch: [10][1/9]	Time 1.092 (1.092)	Data 0.769 (0.769)	Loss 4.7849 (4.7849)	Acc 0.750 (0.750)

    # Batch time: Tempo totale per batch, da feed forward a backprop incluso
    # Data time: Tempo di caricamento dei dati nel batch considerato
    # Loss: funzione di Loss (Cross Entropy) - Sinistra valore corrente, destra media fra tutti i valori
    # Acc: Accuratezza - Sinistra valore corrente, destra media fra tutti i valori
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': current_lr
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\n'.format(epoch,
                                                           i + 1,
                                                           len(data_loader),
                                                           batch_time=batch_time,
                                                           data_time=data_time,
                                                           loss=losses,
                                                           acc=accuracies))

        end_time = time.time()

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': current_lr
    })

    if tb_writer is not None:
        tb_writer.add_scalar('Training/Loss per epoch', losses.avg, epoch)
        tb_writer.add_scalar('Training/Accuracy per epoch', accuracies.avg, epoch)
        tb_writer.add_scalar('Training/Learning Rate per epoch', current_lr, epoch)
