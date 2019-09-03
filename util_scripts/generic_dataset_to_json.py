import argparse
import json
from pathlib import Path

import pandas as pd
import random
import os

from utils_frames import get_n_frames


def convert_splits_to_dict(split_file_path, video_with_label):
    database = {}

    data = pd.read_csv(split_file_path, delimiter=' ', header=None)
    keys = []
    subsets = []
    for i in range(data.shape[0]):
        row = data.iloc[i, :]
        if row[1] == 0:
            continue
        elif row[1] == 1:
            subset = 'training'
        elif row[1] == 2:
            subset = 'validation'

        keys.append(row[0].split('.')[0])
        subsets.append(subset)

    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subsets[i]
        label = video_with_label[data.iloc[i, 0]]
        database[key]['annotations'] = {'label': label}

    return database


def convert_dataset_to_json(output_splits_path, video_dir_path, output_annotations_path, video_with_label, target_classes):
    database = convert_splits_to_dict(output_splits_path, video_with_label)

    dst_data = {'labels': target_classes, 'database': {}}
    dst_data['database'].update(database)

    for k, v in dst_data['database'].items():
        if v['annotations'] is not None:
            label = v['annotations']['label']
        else:
            label = 'test'

        video_path = video_dir_path / label / k
        n_frames = get_n_frames(video_path)
        v['annotations']['segment'] = (1, n_frames + 1)

    with output_annotations_path.open('w+') as dst_file:
        json.dump(dst_data, dst_file)


def generate_split(input_args):
    print("Generating training and validation splits")

    seed = input_args.seed
    split_size_train = input_args.split_size_train
    video_path = input_args.video_path

    is_training = input_args.training
    is_validation = input_args.validation

    random.seed(seed)

    classes_as_folders = os.listdir(video_path)
    video_with_label = {}
    target_classes = []
    result_video_and_subset = []

    for target_class in classes_as_folders:
        target_classes.append(target_class)
        input_video_path = os.path.join(video_path, target_class)
        input_videos = os.listdir(input_video_path)

        for input_video in input_videos:
            if input_video in video_with_label:
                raise ("Duplicate video name {}".format(input_video))
            else:
                video_with_label[input_video] = target_class

            if split_size_train:
                normalized_split = split_size_train / 100
                rand_num = random.uniform(0, 1)
                if rand_num <= normalized_split:
                    result_video_and_subset.append(f"{input_video} 1")  # 1 means training
                else:
                    result_video_and_subset.append(f"{input_video} 2")  # 2 means validation
            else:
                if is_training:
                    result_video_and_subset.append(f"{input_video} 1")  # 1 means training
                elif is_validation:
                    result_video_and_subset.append(f"{input_video} 2")  # 2 means validation
                else:
                    raise ("Must specify --training or --validation or a split_size")

    # [random.randrange(1, 3) for _ in range(10)]

    # print(len(os.listdir('.')))

    # get labels

    return video_with_label, result_video_and_subset, target_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_annotations_path',
                        default=None,
                        type=Path,
                        help='Directory path of the output generic annotation files that will be created')
    parser.add_argument('--seed',
                        default=5,
                        type=int,
                        help='Seed to be used')
    parser.add_argument('--training',
                        action='store_false',
                        help='If provided, will mark all found videos as training videos')
    parser.add_argument('--validation',
                        action='store_false',
                        help='If provided, will mark all found videos as validation videos')
    parser.add_argument('--split_size_train',
                        type=float,
                        help='Determines the split between the training and validation sets')
    parser.add_argument('--video_path',
                        default=None,
                        type=Path,
                        help=(
                            'Path of video directory. Should point to a folder that has subfolders named as the class'))
    parser.add_argument('--output_splits_path',
                        default=None,
                        type=Path,
                        help='Directory path of the output generic split files that will be created')

    args = parser.parse_args()

    video_with_label, list_of_splitted_videos, target_classes = generate_split(args)

    output_splits_path = args.output_splits_path

    with open(output_splits_path, 'w+') as split_file:
        for split in list_of_splitted_videos:
            split_file.write(split)

    output_annotations_path = args.output_annotations_path

    convert_dataset_to_json(output_splits_path, args.video_path, output_annotations_path, video_with_label, target_classes)
