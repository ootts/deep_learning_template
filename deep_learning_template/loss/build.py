from torch.nn import CrossEntropyLoss, SmoothL1Loss

from deep_learning_template.registry import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register('cross_entropy_loss')
def build_cross_entropy_loss(cfg):
    return CrossEntropyLoss()


@LOSS_FUNCTIONS.register('smooth_l1_loss')
def build_cross_entropy_loss(cfg):
    return SmoothL1Loss()


def build_loss_function(cfg):
    return LOSS_FUNCTIONS[cfg.solver.loss_function](cfg)
