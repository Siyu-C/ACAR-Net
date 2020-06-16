# ACAR-Net
Actor-Context-Actor Relation Network for Spatio-temporal Action Localization - 1st place solution in [AVA-Kinetics
Crossover Challenge 2020](https://research.google.com/ava/challenge.html) . 

Code and model will come soon!

| ![Junting Pan][JuntingPan-photo]  | ![Siyu Chen][SiyuChen-photo]  |  ![Zheng Shou][ZhengShou-photo] | ![Jing Shao][JingShao-photo] | ![Hongsheng Li][HongshengLi-photo]  |
|:-:|:-:|:-:|:-:|:-:|
| [Junting Pan][JuntingPan-web]  | [Siyu Chen][SiyuChen-web] | [Zheng Shou][ZhengShou-web] | [Jing Shao][JingShao-web] |  [Hongsheng Li][HongshengLi-web] 

[JuntingPan-web]: https://junting.github.io/
[SiyuChen-web]: https://siyu-c.github.io/
[ZhengShou-web]: http://www.columbia.edu/~zs2262/
[JingShao-web]: https://amandajshao.github.io/
[HongshengLi-web]: https://www.ee.cuhk.edu.hk/~hsli/

[JuntingPan-photo]: https://github.com/Siyu-C/ACAR-Net/blob/master/authors/juntingpan.png "Junting Pan"
[SiyuChen-photo]: https://github.com/Siyu-C/ACAR-Net/blob/master/authors/siyuchen.png "Siyu Chen"
[ZhengShou-photo]: https://github.com/Siyu-C/ACAR-Net/blob/master/authors/zhengshou.png "Zheng Shou"
[JingShao-photo]: https://github.com/Siyu-C/ACAR-Net/blob/master/authors/jingshao.png "JingShao"
[HongshengLi-photo]: https://github.com/Siyu-C/ACAR-Net/blob/master/authors/hongshengli.png "Hongsheng Li"

## Abstract

Localizing persons and recognizing their actions from videos is a challenging task towards high-level video understanding. Recent advances have been achieved by modeling either “actor-actor” or “actorcontext” relations. However, such direct first-order relations are not sufficient for localizing actions in complicated scenes. Some actors might be indirectly related via objects or background context in the scene. Such indirect relations are crucial for determining the action labels but are mostly ignored by existing work. In this paper, we propose to explicitly model the Actor-Context-Actor Relation, which can capture indirect high-order supportive information for effectively reasoning actors’ actions in complex scenes. To this end, we design an Actor-ContextActor Relation Network (ACAR-Net) which builds upon a novel Highorder Relation Reasoning Operator to model indirect relations for spatiotemporal action localization. Moreover, to allow utilizing more temporal contexts, we extend our framework with an Actor-Context Feature Bank for reasoning long-range high-order relations. Extensive experiments on AVA dataset validate the effectiveness of our ACAR-Net. Ablation studies show advantages of modeling high-order relations over existing first-order relation reasoning methods. The proposed ACAR-Net is also the core module of our **1st place solution in AVA-Kinetics
Crossover Challenge 2020**.

## CVPR 2020 AVA-Kinetics Challenge  
Find slides and video presentation of our winning solution on [[Google Slides]](https://docs.google.com/presentation/d/1JrZLddujC2LVl3etUKkbj40o486fnQMzlAHHbc8F9q4/edit?usp=sharing) [[Youtube Video]](https://youtu.be/zJPEmG3LCH4?list=PLw6H4u-XW8siSxqdRVcD5aBn3OTuA7M7x&t=1105) [[Bilibili Video]](https://www.bilibili.com/video/BV1nT4y1J716) (Starting from 18:20).

## Preprint
Find our work on [Arxiv](https://arxiv.org/pdf/2006.07976.pdf).

Please cite with the following Bibtex code:

```
@misc{pan2020actorcontextactor,
    title={Actor-Context-Actor Relation Network for Spatio-Temporal Action Localization},
    author={Junting Pan and Siyu Chen and Zheng Shou and Jing Shao and Hongsheng Li},
    year={2020},
    eprint={2006.07976},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

You may also want to refer to our publication with the more human-friendly Chicago style:

*Junting Pan, Siyu Chen, Zheng Shou, Jing Shao, Hongsheng Li. "Actor-Context-Actor Relation Network for Spatio-Temporal Action Localization." Arxiv 2020.*


## Models
ACAR Net Architecture
![architecture-fig]

[architecture-fig]: https://github.com/Siyu-C/ACAR-Net/blob/master/figs/architecture.png "acar-net architecture"

## Contact
If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/Siyu-C/ACAR-Net/issues) on this github repo. Alternatively, drop us an e-mail at siyuchen@pku.edu.cn and junting.pa@gmail.com .

