# [CVPR 2021] Actor-Context-Actor Relation Network for Spatio-temporal Action Localization
This repository gives the official PyTorch implementation of [Actor-Context-Actor Relation Network for Spatio-temporal Action Localization](https://arxiv.org/pdf/2006.07976.pdf) (CVPR 2021) - 1st place solution of [AVA-Kinetics Crossover Challenge 2020](https://research.google.com/ava/challenge.html).
This codebase also provides a general pipeline for training and evaluation on AVA-style datasets, as well as state-of-the-art action detection models.

| ![Junting Pan][JuntingPan-photo]  | ![Siyu Chen][SiyuChen-photo]  |  ![Zheng Shou][ZhengShou-photo] | ![Jing Shao][JingShao-photo] | ![Hongsheng Li][HongshengLi-photo]  |
|:-:|:-:|:-:|:-:|:-:|
| [Junting Pan][JuntingPan-web]  | [Siyu Chen][SiyuChen-web] | [Zheng Shou][ZhengShou-web] | [Jing Shao][JingShao-web] |  [Hongsheng Li][HongshengLi-web] 

[JuntingPan-web]: https://junting.github.io/
[SiyuChen-web]: https://siyu-c.github.io/
[ZhengShou-web]: http://www.columbia.edu/~zs2262/
[JingShao-web]: https://amandajshao.github.io/
[HongshengLi-web]: https://www.ee.cuhk.edu.hk/~hsli/

[JuntingPan-photo]: https://github.com/Siyu-C/ACAR-Net/blob/master/figs/authors/juntingpan.png "Junting Pan"
[SiyuChen-photo]: https://github.com/Siyu-C/ACAR-Net/blob/master/figs/authors/siyuchen.png "Siyu Chen"
[ZhengShou-photo]: https://github.com/Siyu-C/ACAR-Net/blob/master/figs/authors/zhengshou.png "Zheng Shou"
[JingShao-photo]: https://github.com/Siyu-C/ACAR-Net/blob/master/figs/authors/jingshao.png "JingShao"
[HongshengLi-photo]: https://github.com/Siyu-C/ACAR-Net/blob/master/figs/authors/hongshengli.png "Hongsheng Li"

## Requirements
Some key dependencies are listed below, while others are given in [`requirements.txt`](https://github.com/Siyu-C/ACAR-Net/blob/master/requirements.txt).
- Python >= 3.6
- PyTorch >= 1.3, and a corresponding version of torchvision
- ffmpeg (used in data preparation)
- Download pre-trained models, which are listed in [`pretrained/README.md`](https://github.com/Siyu-C/ACAR-Net/blob/master/pretrained/README.md), to the `pretrained` folder.
- Prepare data. Please refer to [`DATA.md`](https://github.com/Siyu-C/ACAR-Net/blob/master/DATA.md).
- Download annotations files to the `annotations` folder. See [`annotations/README.md`](https://github.com/Siyu-C/ACAR-Net/blob/master/annotations/README.md) for detailed information.

## Usage
Default values for arguments `nproc_per_node`, `backend` and `master_port` are `8`, `nccl` and `31114` respectively.

```
python main.py --config CONFIG_FILE [--nproc_per_node N_PROCESSES] [--backend BACKEND] [--master_addr MASTER_ADDR] [--master_port MASTER_PORT]
```

### Running with Multiple Machines
In this case, the `master_addr` argument must be provided. Moreover, arguments `nnodes` and `node_rank` can be additionally specified (similar to `torch.distributed.launch`), otherwise the program will try to obtain their values from environment variables. See [`distributed_utils.py`](https://github.com/Siyu-C/ACAR-Net/blob/master/distributed_utils.py) for details.

## Model Zoo
Trained models are provided in [`model_zoo/README.md`](https://github.com/Siyu-C/ACAR-Net/blob/master/model_zoo/README.md).

## To-Do List
- Our detections for AVA
- Data preparation for Kinetics dataset, and training on AVA-Kinetics
- Implementation for ACFB

## License
ACAR-Net is released under the [Apache 2.0 license](https://github.com/Siyu-C/ACAR-Net/blob/master/LICENSE).

## CVPR 2020 AVA-Kinetics Challenge  
Find slides and video presentation of our winning solution on [[Google Slides]](https://docs.google.com/presentation/d/1JrZLddujC2LVl3etUKkbj40o486fnQMzlAHHbc8F9q4/edit?usp=sharing) [[Youtube Video]](https://youtu.be/zJPEmG3LCH4?list=PLw6H4u-XW8siSxqdRVcD5aBn3OTuA7M7x&t=1105) [[Bilibili Video]](https://www.bilibili.com/video/BV1nT4y1J716) (Starting from 18:20).

## About Our Paper
Find our work on [arXiv](https://arxiv.org/pdf/2006.07976.pdf).
![architecture-fig]

[architecture-fig]: https://github.com/Siyu-C/ACAR-Net/blob/master/figs/architecture.png "acar-net architecture"

Please cite with the following Bibtex code:

```
@inproceedings{pan2021actor,
  title={Actor-context-actor relation network for spatio-temporal action localization},
  author={Pan, Junting and Chen, Siyu and Shou, Mike Zheng and Liu, Yu and Shao, Jing and Li, Hongsheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={464--474},
  year={2021}
}
```

You may also want to refer to our publication with the more human-friendly Chicago style:

*Pan, Junting, Siyu Chen, Mike Zheng Shou, Yu Liu, Jing Shao, and Hongsheng Li. "Actor-context-actor relation network for spatio-temporal action localization." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 464-474. 2021.*

## Contact
If you have any general question about our work or code which may be of interest to other researchers, please use the [public issues section](https://github.com/Siyu-C/ACAR-Net/issues) of this repository. Alternatively, drop us an e-mail at siyuchen@pku.edu.cn and junting.pa@gmail.com .
