import time
# Removes useless warning when precision, recall or fscore are zero
import warnings

from sklearn.metrics import precision_recall_fscore_support

from utils import AverageMeter, calculate_accuracy, ground_truth_and_predictions

warnings.filterwarnings('ignore', message='(.*)Precision and F-score are ill-defined(.*)')
warnings.filterwarnings('ignore', message='(.*)Recall and F-score are ill-defined(.*)')


def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                class_names,
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

    class_size = len(class_names)
    class_idx = list(range(0, class_size))

    ground_truth_labels = []
    predicted_labels = []

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        ground_truth, predictions = ground_truth_and_predictions(outputs, targets)

        ground_truth_labels.extend(ground_truth)
        predicted_labels.extend(predictions)

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

    losses_avg = losses.avg
    accuracies_avg = accuracies.avg

    precision_epoch, recall_epoch, fscore_epoch, _ = \
        precision_recall_fscore_support(ground_truth_labels,
                                        predicted_labels,
                                        labels=class_idx)

    # print("update")
    # print(f"prec avg:{precs_avg}")
    # print(f"recs_avg: {recs_avg}")
    # print(f"fscores_avg: {fscores_avg}")

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses_avg,
        'acc': accuracies_avg,
        'lr': current_lr,
        'prec': precision_epoch,
        'rec': recall_epoch,
        'fscore': fscore_epoch
    })

    if tb_writer is not None:
        tb_writer.add_scalar('Training/Loss per epoch', losses_avg, epoch)
        tb_writer.add_scalar('Training/Accuracy per epoch', accuracies_avg, epoch)
        tb_writer.add_scalar('Training/Learning Rate per epoch', current_lr, epoch)

        [tb_writer.add_scalar(f"Training/Precision for class {class_names[idx]} per epoch", precision_epoch[idx], epoch) for
         idx in class_idx]
        [tb_writer.add_scalar(f"Training/Recall for class {class_names[idx]} per epoch", recall_epoch[idx], epoch) for idx
         in class_idx]
        [tb_writer.add_scalar(f"Training/F-Score for class {class_names[idx]} per epoch", fscore_epoch[idx], epoch) for
         idx in class_idx]
