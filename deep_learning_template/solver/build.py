from dl_ext.pytorch_ext import OneCycleScheduler
from torch.optim import SGD, Adam
from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    lr = cfg.solver.max_lr
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = cfg.solver.weight_decay
        if "bias" in key:
            lr = cfg.solver.max_lr * cfg.solver.bias_lr_factor
            weight_decay = cfg.solver.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.solver.optimizer == 'SGD':
        optimizer = SGD(params, lr, momentum=cfg.solver.momentum)
    elif cfg.solver.optimizer == 'Adam':
        optimizer = Adam(params, lr)
    else:
        raise NotImplementedError()
    return optimizer


def make_lr_scheduler(cfg, optimizer, max_iter):
    if cfg.solver.scheduler == 'WarmupMultiStepLR':
        return WarmupMultiStepLR(
            optimizer,
            cfg.solver.steps,
            cfg.solver.gamma,
            warmup_factor=cfg.solver.warmup_factor,
            warmup_iters=cfg.solver.warmup_iters,
            warmup_method=cfg.solver.warmup_method,
        )
    elif cfg.solver.scheduler == 'OneCycleScheduler':
        return OneCycleScheduler(
            optimizer,
            cfg.solver.max_lr,
            max_iter
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    else:
        raise NotImplementedError()
