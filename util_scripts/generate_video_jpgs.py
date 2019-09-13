import subprocess
import argparse
from pathlib import Path

from joblib import Parallel, delayed
import os


def video_process(video_file_path, dst_root_path, class_dir_path, existing_videos, fps=-1, size=240):
    ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                   '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                   'stream=width,height,avg_frame_rate,duration').split()
    ffprobe_cmd.append(str(video_file_path))

    p = subprocess.run(ffprobe_cmd, capture_output=True)
    res = p.stdout.decode('utf-8').splitlines()
    if len(res) < 4:
        raise Exception("Video {} is potentially corrupted. Delete it from the input video directory and re-run this "
                        "script.".format(video_file_path))

    str_video = str(video_file_path)
    num_of_dots = str_video.split("/")[-1].count('.')

    if num_of_dots == 0:
        raise Exception("Video {} should have an extension.".format(video_file_path))
    elif num_of_dots > 1:
        print("Video name {} with more than 1 dot. Substituting the other dots with _".format(video_file_path))

        # Slices the string and substitute . with _
        new_video_name = f"{str_video[:str_video.rfind('.')].replace('.', '_')}{str_video[str_video.rfind('.'):]}"
        target_video = os.path.join(class_dir_path, new_video_name)

        os.rename(video_file_path, target_video)
        video_file_path = Path(target_video)

    name = video_file_path.stem

    dst_dir_path = dst_root_path / name

    if name in existing_videos:
        new_video_name = f"{name.split('.')[0]}_{class_dir_path.stem}.{name.split('.')[1]}"
        print("Duplicate video name {}. Substituting with {}".format(dst_dir_path, new_video_name))

        target_video = os.path.join(dst_root_path, new_video_name)
        os.rename(video_file_path, target_video)

        dst_dir_path = target_video
    else:
        existing_videos.append(name)

    if os.path.exists(dst_dir_path):
        # print(f"{dst_dir_path} already exists. Skipping...")
        return
    else:
        os.mkdir(dst_dir_path)

    width = int(res[0])
    height = int(res[1])

    if width > height:
        vf_param = 'scale=-1:{}'.format(size)
    else:
        vf_param = 'scale={}:-1'.format(size)

    if fps > 0:
        vf_param += ',minterpolate={}'.format(fps)

    ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), '-vf', vf_param, '-loglevel', 'error']
    ffmpeg_cmd += ['-threads', '1', '{}/image_%05d.jpg'.format(dst_dir_path)]
    print(f"Writing to {dst_dir_path}")
    subprocess.run(ffmpeg_cmd)


def class_process(class_dir_path, dst_root_path, fps=-1, size=240):
    if not class_dir_path.is_dir():
        return

    dst_class_path = dst_root_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True)

    existing_videos = []
    for video_file_path in sorted(class_dir_path.iterdir()):
        video_process(video_file_path, dst_class_path, class_dir_path, existing_videos, fps, size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path', default=None, type=Path, help='Directory path of videos')
    parser.add_argument(
        'dst_path',
        default=None,
        type=Path,
        help='Directory path of jpg videos')
    parser.add_argument(
        'dataset',
        default='generic',
        type=str,
        help='Dataset name (kinetics | mit | ucf101 | hmdb51 | activitynet | generic)')
    parser.add_argument(
        '--n_jobs', default=-1, type=int, help='Number of parallel jobs')
    parser.add_argument(
        '--fps',
        default=-1,
        type=int,
        help=('Frame rates of output videos. '
              '-1 means original frame rates.'))
    parser.add_argument(
        '--size', default=240, type=int, help='Frame size of output videos.')
    args = parser.parse_args()

    if args.dataset == 'activitynet':
        video_file_paths = [x for x in sorted(args.dir_path.iterdir())]
        status_list = Parallel(
            n_jobs=args.n_jobs,
            backend='threading')(delayed(video_process)(
            video_file_path, args.dst_path, args.fps, args.size)
                                 for video_file_path in video_file_paths)
    else:
        class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]
        test_set_video_path = args.dir_path / 'test'
        if test_set_video_path.exists():
            class_dir_paths.append(test_set_video_path)

        status_list = Parallel(
            n_jobs=args.n_jobs,
            backend='threading')(delayed(class_process)(
            class_dir_path, args.dst_path, args.fps, args.size)
                                 for class_dir_path in class_dir_paths)
