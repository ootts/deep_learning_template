from torch.nn import CrossEntropyLoss

from deep_learning_template.registry import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register('cross_entropy_loss')
def build_cross_entropy_loss(cfg):
    return CrossEntropyLoss()


def build_loss_function(cfg):
    return LOSS_FUNCTIONS[cfg.solver.loss_function](cfg)
