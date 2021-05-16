"""
Adapted code from:
    @inproceedings{hara3dcnns,
      author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
      title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages={6546--6555},
      year={2018},
    }.
"""

import random


class TemporalSampling(object):
    """Temporally sample the given frame indices with a given stride.

    Args:
        step (int, optional): Stride for sampling.
    """

    def __init__(self, step=1):
        self.step = step

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        return frame_indices[::self.step]

    def __repr__(self):
        return '{self.__class__.__name__}(step={self.step})'.format(self=self)


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at the center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
        step (int, optional): Stride when taking the crop.
    """

    def __init__(self, size, step=1):
        self.size = size
        self.step = step

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out[::self.step]

    def __repr__(self):
        return '{self.__class__.__name__}(size={self.size}, step={self.step})'.format(self=self)


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
        step (int, optional): Stride when taking the crop.
    """

    def __init__(self, size, step=1):
        self.size = size
        self.step = step

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out[::self.step]

    def __repr__(self):
        return '{self.__class__.__name__}(size={self.size}, step={self.step})'.format(self=self)
