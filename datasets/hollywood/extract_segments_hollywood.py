import argparse
import csv
import os
import subprocess
import json

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir", help="Videos path.")
parser.add_argument("--annot_file",
                    help="Anotation file path (usually HACS_clips_v1.1.csv)")
parser.add_argument("--output_dir", help="Output path.")
parser.add_argument("--rename_target_class", type=json.loads, help="Optional classes to rename", default={})

FLAGS = parser.parse_args()

video_dir = FLAGS.video_dir
annot_file = FLAGS.annot_file
output_dir = FLAGS.output_dir
rename_target_class = FLAGS.rename_target_class


def _supermakedirs(path, mode):
    if not path or os.path.exists(path):
        return []
    (head, _) = os.path.split(path)
    res = _supermakedirs(head, mode)
    os.mkdir(path)
    os.chmod(path, mode)
    res += [path]
    return res


def mkdir_p(path):
    try:
        _supermakedirs(path, 0o755)  # Supporting Python 2 & 3
    except OSError:  # Python >2.5
        pass


if __name__ == '__main__':
    with open(annot_file, 'r') as loaded_annot_file:
        annotations = loaded_annot_file.readlines()

        file_info = os.path.basename(annot_file).split(".")[0].split("_")
        target_class = file_info[0]
        subset = file_info[1]

        for annotation in annotations:
            annotation = annotation.split("  ")
            video_id = str(annotation[0]).strip()
            positive_example = str(annotation[1]).strip()

            if positive_example == '1':
                input_video_id = "{}.avi".format(video_id)
                input_video_path = os.path.join(video_dir, input_video_id)

                if os.path.exists(input_video_path):
                    if target_class in rename_target_class:
                        target_class = rename_target_class[target_class]

                    output_folder = os.path.join(output_dir, target_class, subset)
                    output_video_path = os.path.join(output_folder, input_video_id)

                    mkdir_p(output_folder)

                    if os.path.exists(output_video_path):
                        print("Output path {} already exists. Skipping...".format(output_video_path))
                    else:
                        command = 'cp %(input_video_path)s %(output_video_path)s' % {
                            'input_video_path': input_video_path,
                            'output_video_path': output_video_path
                        }
                        subprocess.call(command, shell=True)
                else:
                    raise "Input video {} not found.".format(input_video_path)

    loaded_annot_file.close()
