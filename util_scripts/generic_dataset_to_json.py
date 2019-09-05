import argparse
import os
import random
from pathlib import Path
import json

import pandas as pd
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


def convert_dataset_to_json(output_splits_path, frames_dir, video_with_label, target_classes):
    database = convert_splits_to_dict(output_splits_path, video_with_label)

    dst_data = {'labels': list(target_classes), 'database': {}}
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

    result_video_and_subset_train, target_classes_train, video_with_label_train = parse_subset_data(seed, split_size,
                                                                                                    video_path_train,
                                                                                                    True)
    result_video_and_subset_val, target_classes_val, video_with_label_val = parse_subset_data(seed, split_size,
                                                                                              video_path_validation,
                                                                                              False)

    assert len(
        target_classes_train.difference(target_classes_val)) == 0, "The classes should be the same in training and " \
                                                                   "validation set "
    # [random.randrange(1, 3) for _ in range(10)]

    # print(len(os.listdir('.')))

    # get labels

    # Concat dictionary
    return {**video_with_label_train, **video_with_label_val}, \
           result_video_and_subset_train + result_video_and_subset_val, \
           target_classes_train


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
            source_video = os.path.join(input_video_path, input_video)

            if num_of_dots == 0:
                raise ("Video {} should have an extension.".format(input_video))
            elif num_of_dots > 1:
                print("Video name {} with more than 1 dot. Substituting the other dots with _".format(input_video))

                # Slices the string and substitute . with _
                input_video = f"{input_video[:input_video.rfind('.')].replace('.', '_')}{input_video[input_video.rfind('.'):]}"
                target_video = os.path.join(input_video_path, input_video)

                os.rename(source_video, target_video)

            if input_video in videos:
                new_video_name = f"{input_video.split('.')[0]}_{target_class}.{input_video.split('.')[1]}"
                print("Duplicate video name {}. Substituting with {}".format(input_video, new_video_name))

                target_video = os.path.join(input_video_path, new_video_name)
                os.rename(source_video, target_video)
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
    parser.add_argument('--video_path_training',
                        default=None,
                        type=Path,
                        help=(
                            'Path of video directory for training phase. Should point to a folder that has subfolders named as the class'))
    parser.add_argument('--video_path_validation',
                        default=None,
                        type=Path,
                        help=(
                            'Path of video directory for validation phase. Should point to a folder that has subfolders named as the class'))
    parser.add_argument('--output_splits_path',
                        default=None,
                        type=Path,
                        help='Directory path of the output generic split files that will be created')
    parser.add_argument('--frames_dir',
                        default=None,
                        type=Path,
                        help='Directory path of the video frames')

    args = parser.parse_args()

    video_path_train = args.video_path_training
    video_path_validation = args.video_path_validation

    maybe_fix_duplicates(video_path_train)
    maybe_fix_duplicates(video_path_validation)

    video_with_label, list_of_splitted_videos, target_classes = generate_split(args)

    output_splits_path = args.output_splits_path

    with open(output_splits_path, 'w+') as split_file:
        for split in list_of_splitted_videos:
            split_file.write(f"{split}\n")

    frames_dir = args.frames_dir

    json_training = convert_dataset_to_json(output_splits_path, frames_dir, video_with_label,
                                            target_classes)

    json_validation = convert_dataset_to_json(output_splits_path, frames_dir, video_with_label,
                                              target_classes)

    output_annotations_path = args.output_annotations_path

    with output_annotations_path.open('w+') as dst_file:
        data_to_write = {**json_training, **json_validation}
        json.dump(data_to_write, dst_file, indent=4)
