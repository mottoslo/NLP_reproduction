import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import sys

# class Scheduler(_LRScheduler):
#     def __init__(self, 
#                  optimizer: Optimizer,
#                  dim_embed: int,
#                  warmup_steps: int,
#                  last_epoch: int=-1,
#                  verbose: bool=False):

#         self.dim_embed = dim_embed
#         self.warmup_steps = warmup_steps
#         self.num_param_groups = len(optimizer.param_groups)

#         super().__init__(optimizer, last_epoch, verbose)
        
#     def get_lr(self) -> float:
#         lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
#         return [lr] * self.num_param_groups

class TransformerScheduler(_LRScheduler):
    def __init__(self, 
                 optimizer,
                 dim_embed = 512,
                 warmup_steps = 4000,
                 last_epoch = -1,
                 verbose = False):

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer = optimizer, last_epoch = last_epoch)
        
    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups


def calc_lr(step, dim_embed, warmup_steps):
    return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))

if __name__ == '__main__':
    test1 = calc_lr(4001,512,4000)
    print(test1)
