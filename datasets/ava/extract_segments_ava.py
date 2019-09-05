import argparse
import csv
import os
import subprocess
import json

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir", help="Videos path.")
parser.add_argument("--annot_file",
                    help="Anotation file path (usually ava_train_v2.2.csv)")
parser.add_argument("--output_dir", help="Output path.")
parser.add_argument("--filter_on_class", help="Class to filter on. Mandatory", type=str)
parser.add_argument("--rename_target_class", type=json.loads, help="Optional target class to rename", default={})

FLAGS = parser.parse_args()

filter_on_class = FLAGS.filter_on_class
video_dir = FLAGS.video_dir
annot_file = FLAGS.annot_file
output_dir = FLAGS.output_dir
rename_target_class = FLAGS.rename_target_class

def hou_min_sec(millis):
    millis = int(millis)
    seconds = (millis / 1000) % 60
    seconds = int(seconds)
    minutes = (millis / (1000 * 60)) % 60
    minutes = int(minutes)
    hours = millis / (1000 * 60 * 60)

    msecs = millis % 1000
    return "%d:%d:%d.%d" % (hours, minutes, seconds, msecs)


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


def jump_to_end_segment(rows, current_idx, target_label, prev_video_id, prev_start_time_seconds, res):
    tmp_counter = current_idx + 1

    # Look ahead if there are some left frames
    list_of_next_records, skipped_rows = recursive_search(prev_start_time_seconds, rows, tmp_counter, prev_video_id,
                                                          target_label)
    # assert len(list_of_next_records) == 0 or len(list_of_next_records) == 1, "The resulting list of records should have" \
    #                                                                          "length 1 or 0. {}".format(list_of_next_records)

    res.append((list_of_next_records, skipped_rows))

    if list_of_next_records:
        # Recursive
        return jump_to_end_segment(rows, current_idx + 1, target_label, prev_video_id, prev_start_time_seconds + 1, res)
    else:
        return res

    # while inner_counter < len(rows):
    #
    #     video_id_inner = str(next_row[0])
    #     start_time_seconds_inner = float(next_row[1])
    #     label_inner = str(next_row[6])
    #
    #     if label_inner == target_label and prev_video_id == video_id_inner:
    #         seconds_to_trim = start_time_seconds_inner
    #         inner_counter += 1
    #     else:
    #         break
    #
    # rows_to_skip_inner = inner_counter - current_idx


def recursive_search(prev_start_time_seconds, rows, tmp_counter, prev_video_id, target_label):
    tmp_list_next_records = []
    current_time_skipped = 0

    while tmp_counter < len(rows):
        next_tmp_row = rows[tmp_counter]
        next_time_sec = float(next_tmp_row[1])

        if next_time_sec == prev_start_time_seconds:
            tmp_counter += 1
            current_time_skipped += 1
            continue
        elif next_time_sec == (prev_start_time_seconds + 1):
            tmp_list_next_records.append(next_tmp_row)
            tmp_counter += 1
        else:
            break

    return ([x for x in tmp_list_next_records if str(x[0]) == prev_video_id and str(x[6]) == target_label],
            current_time_skipped)


if __name__ == '__main__':
    with open(annot_file, 'r') as loaded_annot_file:
        csvreader = csv.reader(loaded_annot_file, delimiter=',')

        rows_to_skip = 0
        row_list = list(csvreader)
        counter = 0

        while counter < len(row_list):
            if rows_to_skip != 0:
                rows_to_skip -= 1
                counter += 1
                continue

            current_row = row_list[counter]
            video_id = str(current_row[0])
            start_time_seconds = float(current_row[1])
            label = str(current_row[6])

            if label == filter_on_class:
                # the rows to skip equals the segments (of 1 second) taken
                result = []
                result = jump_to_end_segment(row_list, counter, label, video_id, start_time_seconds, result)

                end_sec_video = start_time_seconds + len(result)
                rows_to_skip = sum(n for _, n in result)

                if os.path.exists(os.path.join(video_dir, "{}.mp4".format(video_id))):
                    input_video = os.path.join(video_dir, "{}.mp4".format(video_id))
                    ext = '.mp4'
                elif os.path.exists(os.path.join(video_dir, "{}.webm".format(video_id))):
                    input_video = os.path.join(video_dir, "{}.webm".format(video_id))
                    ext = '.webm'
                elif os.path.exists(os.path.join(video_dir, "{}.mkv".format(video_id))):
                    input_video = os.path.join(video_dir, "{}.mkv".format(video_id))
                    ext = '.mkv'
                else:
                    raise ("No video found: {} in input video directory: {}".format(video_id, video_dir))

                if label in rename_target_class:
                    label = rename_target_class[label]

                output_video_folder = os.path.join(output_dir, label)
                output_video = os.path.join(output_video_folder, "{}_{}_{}.{}".format(video_id,
                                                                                      start_time_seconds,
                                                                                      end_sec_video,
                                                                                      ext))

                mkdir_p(output_video_folder)

                if os.path.exists(input_video):
                    if os.path.exists(output_video):
                        print("Output path {} already exists. Skipping...".format(output_video))
                    else:
                        ffmpeg_command = 'ffmpeg -ss %(start_timestamp)s -i \
                        %(videopath)s -g 1 -force_key_frames 0 \
                        -t %(clip_length)d -loglevel error %(outpath)s' % {
                            'start_timestamp': hou_min_sec(start_time_seconds * 1000),
                            # 'end_timestamp': hou_min_sec(clip_end * 1000),
                            'clip_length': (end_sec_video - start_time_seconds),
                            'videopath': input_video,
                            'outpath': output_video}

                        subprocess.call(ffmpeg_command, shell=True)
                else:
                    print("The input video: {} does not exists or ".format(input_video))

            counter += 1

    loaded_annot_file.close()
