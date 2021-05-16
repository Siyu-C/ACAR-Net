import argparse
import math
import multiprocessing
import os

parser = argparse.ArgumentParser(description='Frame Extraction')
parser.add_argument('--video_dir', type=str, required=True)
parser.add_argument('--frame_dir', type=str, required=True)
parser.add_argument('--num_processes', type=int, default=64)
args = parser.parse_args()


def main(video_file, frame_rate=30, clip_start=900, clip_end=1800):
    video_path = os.path.join(args.video_dir, video_file)
   
    video_id = video_file.rsplit('.', 1)[0]
    frame_root = os.path.join(args.frame_dir, video_id)
    os.makedirs(frame_root, exist_ok=True)
    
    '''EXTRACT FRAMES WITH FFMPEG'''
    try:
        frame_path = os.path.join(frame_root, 'image_%06d.jpg')
        ffmpeg_command = 'ffmpeg -ss {} -i {} -qscale:v 1 -frames:v {} -vf "scale=-1:340,fps={}" {} -loglevel error < /dev/null'.format(
            clip_start,
            video_path,
            math.ceil((clip_end - clip_start) * frame_rate + 1e-8),
            frame_rate,
            frame_path,
        )
        ret = os.system(ffmpeg_command)
        assert ret == 0, 'frame extraction failed with exit value {}'.format(ret // 256)
        print('[SUCCESS] ({})'.format(video_file), flush=True)
    except BaseException as e:
        print('[ERROR] ({}) {}'.format(video_file, str(e).replace('\n', ' ')), flush=True)


if __name__ == '__main__':
    videos = os.listdir(args.video_dir)

    with multiprocessing.Pool(args.num_processes) as p:
        p.map(main, videos)
