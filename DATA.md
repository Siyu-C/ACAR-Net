# Data Preparation

## AVA

First, download AVA training and validation videos. An example script can be found at [`download_videos.sh`](https://github.com/facebookresearch/video-long-term-feature-banks/blob/master/dataset_tools/ava/download_videos.sh).

Then, extract frames between 15th and 30th minutes. You may use our implementation at [`tools/extract_frames.py`](https://github.com/Siyu-C/ACAR-Net/blob/master/tools/extract_frames.py). To execute it, use the following command, which will save frames to the `data` folder:
```
python tools/extract_frames.py --video_dir YOUR_VIDEO_DIRECTORY --frame_dir data [--num_processes NUM_PROCESSES]
```

The final `data` folder should have the following structure:
```
data
|_ [AVA train_val video name 0]
|  |_ image_000001.jpg
|  |_ image_000002.jpg
|  |_ ...
|_ [AVA train_val video name 1]
|  |_ image_000001.jpg
|  |_ image_000002.jpg
|  |_ ...
|_ ...
|_ [AVA train_val video name 298]
   |_ image_000001.jpg
   |_ image_000002.jpg
   |_ ...
```
