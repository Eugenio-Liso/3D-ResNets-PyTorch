import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import csv


def load_csv(input_csv_path, row, column, plot_number, titleName):
    plt.subplot(row, column, plot_number)

    with open(input_csv_path, 'r') as loaded_file:
        csvreader = csv.reader(loaded_file, delimiter='\t')
        next(csvreader, None)  # Skip headers

        losses = []
        accuracies = []
        epochs = []
        for trainval_log in csvreader:
            epoch = int(trainval_log[0])
            loss = float(trainval_log[1])
            acc = float(trainval_log[2])

            losses.append(loss)
            accuracies.append(acc)
            epochs.append(epoch)

        plt.plot(epochs, losses, label="Loss", marker=".")
        plt.plot(epochs, accuracies, label="Accuracy", linestyle="--")

        plt.title(titleName)
        plt.ylabel('Loss and accuracy')
        plt.xlabel('Epocs')

        plt.legend(loc="upper right")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_log',
                        required=True,
                        type=Path,
                        help='Path to the train.log')

    parser.add_argument('--val_log',
                        required=True,
                        type=Path,
                        help='Path to the val.log')

    args = parser.parse_args()

    input_csv_train = args.train_log
    input_csv_val = args.val_log

    _ = plt.figure("Training/Validation loss and accuracy")

    load_csv(input_csv_train, 1, 2, 1, "Training loss and accuracy")
    load_csv(input_csv_val, 1, 2, 2, "Validation loss and accuracy")
    plt.show()
