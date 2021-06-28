from bisect import bisect_right
import math
import torch


def get_scheduler(config, optimizer, n_epochs, epoch_steps=1):
    if config.type == 'step':
        return StepLRScheduler(
            optimizer=optimizer, 
            milestones=[int(_ * epoch_steps) for _ in config.milestone_epochs], 
            lr_mults=config.lr_mults, 
            base_lr=config.base_lr, 
            warmup_lr=config.warmup_lr, 
            warmup_steps=int(config.warmup_epochs * epoch_steps)
        )
    elif config.type == 'cosine':
        return CosineLRScheduler(
            optimizer=optimizer, 
            T_max=n_epochs * epoch_steps, 
            eta_min=config.min_lr, 
            base_lr=config.base_lr, 
            warmup_lr=config.warmup_lr, 
            warmup_steps=int(config.warmup_epochs * epoch_steps)
        )
    else:
        raise RuntimeError('unknown lr_scheduler type: {}'.format(config.type))


class _LRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_iter = last_iter

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def _get_new_lr(self):
        raise NotImplementedError

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            param_group['lr'] = lr


class _WarmUpLRScheduler(_LRScheduler):

    def __init__(self, optimizer, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        super(_WarmUpLRScheduler, self).__init__(optimizer, last_iter)
    
    def _get_warmup_lr(self):
        if self.warmup_steps > 0 and self.last_iter < self.warmup_steps:
            # first compute relative scale for self.base_lr, then multiply to base_lr
            scale = ((self.last_iter/self.warmup_steps)*(self.warmup_lr - self.base_lr) + self.base_lr)/self.base_lr
            return [scale * base_lr for base_lr in self.base_lrs]
        else:
            return None


class StepLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, milestones, lr_mults, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        super(StepLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)

        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestone, lr_mults)
        for x in milestones:
            assert isinstance(x, int)
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.lr_mults = [1.0]
        for x in lr_mults:
            self.lr_mults.append(self.lr_mults[-1]*x)
    
    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        pos = bisect_right(self.milestones, self.last_iter)
        scale = self.warmup_lr*self.lr_mults[pos] / self.base_lr
        return [base_lr*scale for base_lr in self.base_lrs]


class CosineLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, T_max, eta_min, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        super(CosineLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)
        self.T_max = T_max
        self.eta_min = eta_min

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        step_ratio = (self.last_iter-self.warmup_steps) / (self.T_max-self.warmup_steps)
        target_lr = self.eta_min + (self.warmup_lr - self.eta_min)*(1 + math.cos(math.pi * step_ratio)) / 2
        scale = target_lr / self.base_lr
        return [scale*base_lr for base_lr in self.base_lrs]
