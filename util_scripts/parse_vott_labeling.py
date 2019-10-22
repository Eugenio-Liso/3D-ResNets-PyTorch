import argparse
import csv
import os
import subprocess
from pathlib import Path


def hou_min_sec(millis):
    millis = int(millis)
    seconds = (millis / 1000) % 60
    seconds = int(seconds)
    minutes = (millis / (1000 * 60)) % 60
    minutes = int(minutes)
    hours = millis / (1000 * 60 * 60)
    hours = int(hours)

    msecs = millis % 1000
    return "%s:%s:%s.%s" % (str(hours).zfill(2), str(minutes).zfill(2), str(seconds).zfill(2), str(msecs).zfill(3))


"""
This script works only if you tag video clips with two 'delimiters': The first 'starting' one must be labeled with
the target_class, for example running. The second 'ending' one must be labeled with another name, not contained in
the possible classes, for example end_block
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vott_csv', type=Path, help='Path of the CSV produced by Microsoft VOTT tool v2.1.0',
                        required=True)
    parser.add_argument('--input_video_dir', default=None, type=Path, required=True,
                        help='Directory containing the videos labeled by VOTT')
    parser.add_argument('--output_frames_dir', default=None, type=Path, required=True,
                        help='Directory path of the output frames, according to the input CSV containing video '
                             'labelings')
    args = parser.parse_args()

    vott_csv = args.vott_csv
    output_frames_dir = args.output_frames_dir
    input_video_dir = args.input_video_dir

    with open(vott_csv, 'r') as loaded_vott_csv:
        csvreader = csv.reader(loaded_vott_csv, delimiter=',')

        vott_records = list(csvreader)
        counter = 1
        target_classes = set()

        while counter < len(vott_records):

            vott_record = vott_records[counter]
            dummy_record = vott_records[counter + 1]
            counter += 2

            image_info = vott_record[0].split('#')

            video_name = image_info[0]
            input_video_path = os.path.join(input_video_dir, video_name)

            target_class = vott_record[5]
            target_classes.add(target_class)

            if dummy_record[5] in target_classes:
                raise Exception(
                    "Every labeled video clip must have a starting record (with the target class) and a dummy"
                    "record, indicating the end of the clip")

            num_of_dots = video_name.count('.')

            if num_of_dots == 0:
                raise Exception("Video {} should have an extension.".format(video_name))
            elif num_of_dots > 1:

                # Slices the string and substitute . with _
                new_video_name = f"{video_name[:video_name.rfind('.')].replace('.', '_')}{video_name[video_name.rfind('.'):]}"
                target_video = os.path.join(input_video_dir, new_video_name)

                if not os.path.exists(target_video):
                    print("Video name {} with more than 1 dot. Substituting the other dots with _".format(video_name))
                    os.rename(input_video_path, target_video)
                video_name = new_video_name
                input_video_path = os.path.join(input_video_dir, video_name)

            start_seconds = hou_min_sec(float(image_info[1].split('=')[1]) * 1000)
            end_seconds = hou_min_sec(float(dummy_record[0].split('#')[1].split('=')[1]) * 1000)

            output_frames_subdir = os.path.join(output_frames_dir,
                                                target_class,
                                                f"{video_name.split('.')[0]}_{start_seconds.replace('.', '_')}_{end_seconds.replace('.', '_')}")
            output_frames_path = os.path.join(output_frames_subdir, "image_%05d.jpg")

            if os.path.exists(output_frames_subdir):
                print(f"{output_frames_subdir} already exists. Skipping...")
                continue
            else:
                os.makedirs(output_frames_subdir, mode=0o755)

            ffmpeg_command = 'ffmpeg -ss %(start_timestamp)s -i %(videopath)s -to %(clip_length)s \
                -copyts -loglevel error %(outpath)s' % {
                'start_timestamp': start_seconds,
                'clip_length': end_seconds,
                'videopath': input_video_path,
                'outpath': output_frames_path}

            print(f'Extracting frames for video {input_video_path} to {output_frames_subdir}')

            subprocess.call(ffmpeg_command, shell=True)
