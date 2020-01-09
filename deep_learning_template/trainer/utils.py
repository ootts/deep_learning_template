from enum import IntEnum

from dl_ext.pytorch_ext import get_rank


def to_cuda(x):
    if hasattr(x, 'cuda'):
        return x.cuda(device=get_rank())
    elif isinstance(x, (list, tuple)):
        return [to_cuda(xi) for xi in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}


def to_cpu(x):
    if hasattr(x, 'cpu'):
        return x.cpu()
    elif isinstance(x, (list, tuple)):
        return [to_cpu(xi) for xi in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}


def batch_gpu(batch):
    x, y = batch
    return to_cuda(x), to_cuda(y)


def format_time(t):
    t = int(t)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    if h != 0:
        return f'{h}:{m:02d}:{s:02d}'
    else:
        return f'{m:02d}:{s:02d}'


class TrainerState(IntEnum):
    BASE = 1
    PARALLEL = 2
    DISTRIBUTEDPARALLEL = 3


def split_list(vals, skip_start: int, skip_end: int):
    return vals[skip_start:-skip_end] if skip_end > 0 else vals[skip_start:]
