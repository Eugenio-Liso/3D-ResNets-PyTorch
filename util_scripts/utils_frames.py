from subprocess import check_output


def get_n_frames(video_path):
    return len([
        x for x in video_path.iterdir()
        if 'image' in x.name and x.name[0] != '.'
    ])


def get_fps(video_file_path):
    ffprobe_command = \
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of',
         'default=nokey=1:noprint_wrappers=1', "{}".format(
            video_file_path)]

    fps = int(check_output(ffprobe_command).decode("utf-8").strip())

    return fps
