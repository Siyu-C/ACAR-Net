import os
import csv
import logging
import math
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

_LOGGER = None


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=" * 105,
    ]

    row_format = "{name:<60} {shape:>27} ={total_size:>15,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 105)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


def create_logger(log_file, level=logging.INFO):
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    l = logging.getLogger('global')
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    l.propagate = False
    _LOGGER = l
    return l


def get_logger():
    return _LOGGER


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


class AverageMeter(object):
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history, self.history_num = [], []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        assert num > 0
        if self.length > 0:
            self.history.append(val * num)
            self.history_num.append(num)
            if len(self.history) > self.length:
                del self.history[0]
                del self.history_num[0]

            self.val = val
            self.avg = np.sum(self.history) / np.sum(self.history_num)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


class DistributedSampler(Sampler):
    def __init__(self, dataset, world_size=None, rank=None, round_down=False):
        if world_size is None:
            world_size = get_world_size()
        if rank is None:
            rank = get_rank()
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.round_down = round_down
        self.epoch = 0

        self.total_size = len(self.dataset)
        if self.round_down:
            self.num_samples = int(math.floor(len(self.dataset) / self.world_size))
        else:
            self.num_samples = int(math.ceil(len(self.dataset) / self.world_size))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))

        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        if self.round_down:
            assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def load_pretrain(pretrain_opt, net):
    checkpoint = torch.load(pretrain_opt.path, map_location=lambda storage, loc: storage.cuda())
    if pretrain_opt.get('state_dict_key', None) is not None:
        checkpoint = checkpoint[pretrain_opt.state_dict_key]

    if pretrain_opt.get('delete_prefix', None):
        keys = set(checkpoint.keys())
        for k in keys:
            if k.startswith(pretrain_opt.delete_prefix):
                checkpoint.pop(k)
    if pretrain_opt.get('replace_prefix', None) is not None:
        keys = set(checkpoint.keys())
        for k in keys:
            if k.startswith(pretrain_opt.replace_prefix):
                new_k = pretrain_opt.get('replace_to', '') + k[len(pretrain_opt.replace_prefix):]
                checkpoint[new_k] = checkpoint.pop(k)
    net.load_state_dict(checkpoint, strict=False)

    if get_rank() == 0:
        ckpt_keys = set(checkpoint.keys())
        own_keys = set(net.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        ignore_keys = ckpt_keys - own_keys
        loaded_keys = own_keys - missing_keys

        logger = get_logger()
        for k in missing_keys:
            logger.info('Caution: missing key {}'.format(k))
        for k in ignore_keys:
            logger.info('Caution: redundant key {}'.format(k))
        logger.info('Loaded {} key(s) from pre-trained model at {}'.format(len(loaded_keys), pretrain_opt.path))
