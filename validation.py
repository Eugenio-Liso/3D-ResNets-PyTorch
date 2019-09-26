import torch
import time
import sys

from utils import AverageMeter, calculate_accuracy, calculate_precision_and_recall


def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              logger,
              tb_writer=None):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    precs = AverageMeter()
    recs = AverageMeter()
    fscores = AverageMeter()

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            prec, rec, fscore = calculate_precision_and_recall(outputs, targets)
            precs.update(prec, inputs.size(0))
            recs.update(rec, inputs.size(0))
            fscores.update(fscore, inputs.size(0))

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))

            end_time = time.time()

    losses_avg = losses.avg
    accuracies_avg = accuracies.avg

    precs_avg = precs.avg
    recs_avg = recs.avg
    fscores_avg = fscores.avg

    logger.log({
        'epoch': epoch,
        'loss': losses_avg,
        'acc': accuracies_avg,
        'prec': precs_avg,
        'rec': recs_avg,
        'fscore': fscores_avg
    })

    if tb_writer is not None:
        tb_writer.add_scalar('Validation/Loss per epoch', losses_avg, epoch)
        tb_writer.add_scalar('Validation/Accuracy per epoch', accuracies_avg, epoch)

        tb_writer.add_scalar('Validation/Precision per epoch', precs_avg, epoch)
        tb_writer.add_scalar('Validation/Recall per epoch', recs_avg, epoch)
        tb_writer.add_scalar('Validation/F-Score per epoch', fscores_avg, epoch)

    return losses.avg
