import argparse
import csv
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir", help="Videos path.")
parser.add_argument("--annot_file",
                    help="Anotation file path (usually HACS_clips_v1.1.csv)")
parser.add_argument("--output_dir", help="Output path.")
parser.add_argument("--filter_on_classes", nargs='*', help="Optional classes to filter on", default=[])

FLAGS = parser.parse_args()

filter_on_classes = FLAGS.filter_on_classes
video_dir = FLAGS.video_dir
annot_file = FLAGS.annot_file
output_dir = FLAGS.output_dir


def hou_min_sec(millis):
    millis = int(millis)
    seconds = (millis / 1000) % 60
    seconds = int(seconds)
    minutes = (millis / (1000 * 60)) % 60
    minutes = int(minutes)
    hours = int(millis / (1000 * 60 * 60))
    return "%d:%d:%d" % (hours, minutes, seconds)


if __name__ == '__main__':
    with open(annot_file, 'r') as loaded_annot_file:
        csvreader = csv.reader(loaded_annot_file, delimiter=',')

        next(csvreader, None) # Skip headers

        for annotation in csvreader:
            target_class = str(annotation[0]).replace(' ', '_')

            # If it is empty or we are filtering on that class
            if not filter_on_classes or (target_class in filter_on_classes):
                video_id = str(annotation[1]).replace(' ', '_')
                # subset = str(annotation[2])
                start_time_seconds = float(annotation[3])
                end_time_seconds = float(annotation[4])
                label = str(annotation[5])

                if label == '1': # && subset = 'training' or subset = 'validation' FOR NOW I TAKE ALL THE VIDEOS AND DISCARD NEGATIVE EXAMPLES
                    input_video = os.path.join(video_dir, target_class, "v_{}.mp4".format(video_id))
                    output_video = os.path.join(output_dir, target_class, "v_{}_{}_{}.mp4".format(video_id,
                                                                                                  start_time_seconds,
                                                                                                  end_time_seconds))

                    ffmpeg_command = 'rm %(outpath)s > /dev/null;  \
                                          ffmpeg -ss %(start_timestamp)s -i \
                                          %(videopath)s -g 1 -force_key_frames 0 \
                                          -t %(clip_length)d -loglevel error %(outpath)s' % {
                        'start_timestamp': hou_min_sec(start_time_seconds * 1000),
                        # 'end_timestamp': hou_min_sec(clip_end * 1000),
                        'clip_length': end_time_seconds,
                        'videopath': input_video,
                        'outpath': output_video}

                    subprocess.call(ffmpeg_command, shell=True)

    loaded_annot_file.close()
