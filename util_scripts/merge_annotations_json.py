import argparse
import json
from pathlib import Path


def check_for_duplicates(databases):
    existing_videos = []
    for database in databases:
        for video_key in database.keys():
            if video_key in existing_videos:
                raise Exception("Video named {} cannot be merged because it is duplicated. Please, rename this video "
                                "in the annotation json AND in the frames directory".format(video_key))
            else:
                existing_videos.append(video_key)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--annotations_paths', nargs='+', help='Paths of various annotations json to merge',
                        required=True)
    parser.add_argument('--output_merged_annotations',
                        default=None,
                        type=Path,
                        help='Directory path of the output generic annotation files that will be created '
                             'after the merge',
                        required=True)
    args = parser.parse_args()

    input_json_annotations = []

    output_merged_annotations = args.output_merged_annotations
    annotations_paths = args.annotations_paths

    for input_json_file_path in annotations_paths:
        # Read JSON file
        with open(input_json_file_path) as data_file:
            data_loaded = json.load(data_file)
            input_json_annotations.append(data_loaded)

    target_classes_size = 0
    target_classes = set()
    databases = []
    for input_json in input_json_annotations:
        labels = input_json['labels']

        for label in labels:
            target_classes.update(label)

        database = input_json['database']
        databases.append(database)

    check_for_duplicates(databases)
    merged_db = {}
    [merged_db.update({**db}) for db in databases]

    print(merged_db)
    output_merge = {'labels': sorted(list(target_classes)), 'database': merged_db}

    with output_merged_annotations.open('w+') as dst_file:
        json.dump(output_merge, dst_file, indent=4)
