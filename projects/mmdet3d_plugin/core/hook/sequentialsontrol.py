# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
#from mmdet3d.core.hook.utils import is_parallel
from torch import nn

__all__ = ['SequentialControlHook']


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)



@HOOKS.register_module()
class SequentialControlHook(Hook):
    """ """

    def __init__(self, temporal_start_epoch=1):
        super().__init__()
        self.temporal_start_epoch=temporal_start_epoch

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.use_temporal=flag
        else:
            runner.model.module.use_temporal = flag

    def before_run(self, runner):
        self.set_temporal_flag(runner, False)

    def before_train_epoch(self, runner):
        if runner.epoch >= self.temporal_start_epoch:
            self.set_temporal_flag(runner, True)