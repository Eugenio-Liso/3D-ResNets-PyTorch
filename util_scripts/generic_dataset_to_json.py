import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd
from utils_frames import get_n_frames


def convert_splits_to_dict(split_file_path, video_with_label, augmented_data):
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

        if augmented_data:
            keys.append(row[0])  # Augmented data will not have any extension in its name
        else:
            keys.append(row[0].split('.')[0])
        subsets.append(subset)

    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subsets[i]
        label = video_with_label[data.iloc[i, 0]]
        database[key]['annotations'] = {'label': label}

    return database


def convert_dataset_to_json(output_splits_path, frames_dir, video_with_label, target_classes, augmented_data):
    database = convert_splits_to_dict(output_splits_path, video_with_label, augmented_data)

    dst_data = {'labels': sorted(list(target_classes)), 'database': {}}
    dst_data['database'].update(database)

    for k, v in dst_data['database'].items():
        if v['annotations'] is not None:
            label = v['annotations']['label']
        else:
            label = 'test'

        video_path = frames_dir / label / k
        n_frames = get_n_frames(video_path)
        v['annotations']['segment'] = (1, n_frames + 1)

    return dst_data


def generate_split(input_args):
    print("Generating training and validation splits")

    seed = input_args.seed
    split_size = input_args.split_size_train
    video_path_train = input_args.video_path_training
    video_path_validation = input_args.video_path_validation
    video_path_trainval = args.video_path_trainval
    augmented_data = args.augmented_data

    if augmented_data:
        result_video_and_subset_aug, target_classes_aug, video_with_label_aug = \
            parse_subset_data(seed,
                              split_size,
                              args.frames_dir,  # Changing the input directory for augmented data
                              None)  # Unused

        return video_with_label_aug, \
               result_video_and_subset_aug, \
               target_classes_aug

    elif video_path_train and video_path_validation:
        result_video_and_subset_train, target_classes_train, video_with_label_train = parse_subset_data(seed,
                                                                                                        split_size,
                                                                                                        video_path_train,
                                                                                                        True)
        result_video_and_subset_val, target_classes_val, video_with_label_val = parse_subset_data(seed, split_size,
                                                                                                  video_path_validation,
                                                                                                  False)
        assert len(
            target_classes_train.difference(target_classes_val)) == 0, "The classes should be the same in training and " \
                                                                       "validation set "
        # Concat dictionary
        return {**video_with_label_train, **video_with_label_val}, \
               result_video_and_subset_train + result_video_and_subset_val, \
               target_classes_train
    else:
        result_video_and_subset_trainval, target_classes_trainval, video_with_label_trainval = \
            parse_subset_data(seed,
                              split_size,
                              video_path_trainval,
                              None)  # Unused

        return video_with_label_trainval, \
               result_video_and_subset_trainval, \
               target_classes_trainval

    # [random.randrange(1, 3) for _ in range(10)]

    # print(len(os.listdir('.')))

    # get labels


def parse_subset_data(seed, split_size, video_path, is_training):
    random.seed(seed)
    classes_as_folders = os.listdir(video_path)
    video_with_label = {}
    target_classes = set()

    result_video_and_subset = []
    for target_class in classes_as_folders:
        target_classes.add(target_class)
        input_video_path = os.path.join(video_path, target_class)
        input_videos = os.listdir(input_video_path)

        for input_video in input_videos:
            if input_video in video_with_label:
                raise ("Duplicate video name {}".format(input_video))
            else:
                video_with_label[input_video] = target_class

            if split_size:
                rand_num = random.uniform(0, 1)
                if rand_num <= split_size:
                    result_video_and_subset.append(f"{input_video} 1")  # 1 means training
                else:
                    result_video_and_subset.append(f"{input_video} 2")  # 2 means validation
            else:
                if is_training is None:
                    raise Exception("When using the --video_path_trainval or --frames_dir with augmented data, "
                                    "you must specify the split size")
                if is_training:
                    result_video_and_subset.append(f"{input_video} 1")  # 1 means training
                else:
                    result_video_and_subset.append(f"{input_video} 2")  # 2 means validation
    return result_video_and_subset, target_classes, video_with_label


def maybe_fix_duplicates(video_path):
    classes_as_folders = os.listdir(video_path)
    videos = []

    for target_class in classes_as_folders:
        input_video_path = os.path.join(video_path, target_class)
        input_videos = os.listdir(input_video_path)

        for input_video in input_videos:
            num_of_dots = input_video.count('.')

            if num_of_dots == 0:
                raise Exception("Video {} should have an extension.".format(input_video))
            elif num_of_dots > 1:
                raise Exception(
                    "Video name {} with more than 1 dot. Substituting the other dots with _".format(input_video))
            if input_video in videos:
                raise Exception(f"{input_video} is duplicated. Please change its name.")
            else:
                videos.append(input_video)


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
    parser.add_argument('--split_size_train',
                        type=float,
                        help='Determines the split between the training and validation sets')
    parser.add_argument('--video_path_trainval',
                        default=None,
                        type=Path,
                        help=(
                            'Path of video directory for training AND validation phase. Should point to a folder that has subfolders named as the class. '
                            'Should NOT be used in conjunction with video_path_training and video_path_validation'))

    parser.add_argument('--video_path_training',
                        default=None,
                        type=Path,
                        help=(
                            'Path of video directory for training phase. Should point to a folder that has subfolders named as the class.'
                            'Sould NOT be used in conjunction with video_path_trainval'))
    parser.add_argument('--video_path_validation',
                        default=None,
                        type=Path,
                        help=(
                            'Path of video directory for validation phase. Should point to a folder that has subfolders named as the class.'
                            'Sould NOT be used in conjunction with video_path_trainval'))
    parser.add_argument('--output_splits_path',
                        default=None,
                        type=Path,
                        help='Directory path of the output generic split files that will be created')
    parser.add_argument('--frames_dir',
                        default=None,
                        type=Path,
                        help='Directory path of the video frames')
    parser.add_argument('--augmented_data',
                        action='store_true',
                        help='This setting should be used ONLY when the input data is augmented WITHOUT any --video_path_* '
                             'parameter')

    args = parser.parse_args()

    video_path_train = args.video_path_training
    video_path_validation = args.video_path_validation
    video_path_trainval = args.video_path_trainval
    augmented_data = args.augmented_data

    if not augmented_data:
        if video_path_train and video_path_validation:
            maybe_fix_duplicates(video_path_train)
            maybe_fix_duplicates(video_path_validation)
        elif video_path_trainval:
            maybe_fix_duplicates(video_path_trainval)
        else:
            raise ("A video input path must be specified.")

    video_with_label, list_of_splitted_videos, target_classes = generate_split(args)

    output_splits_path = args.output_splits_path

    with open(output_splits_path, 'w+') as split_file:
        for split in list_of_splitted_videos:
            split_file.write(f"{split}\n")

    frames_dir = args.frames_dir
    output_annotations_path = args.output_annotations_path

    if video_path_train and video_path_validation:
        json_training = convert_dataset_to_json(output_splits_path, frames_dir, video_with_label,
                                                target_classes, augmented_data)

        json_validation = convert_dataset_to_json(output_splits_path, frames_dir, video_with_label,
                                                  target_classes, augmented_data)
        with output_annotations_path.open('w+') as dst_file:
            data_to_write = {**json_training, **json_validation}
            json.dump(data_to_write, dst_file, indent=4)
    else:
        json_trainval = convert_dataset_to_json(output_splits_path, frames_dir, video_with_label,
                                                target_classes, augmented_data)
        with output_annotations_path.open('w+') as dst_file:
            json.dump(json_trainval, dst_file, indent=4)
