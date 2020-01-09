from typing import Dict

from torch import Tensor

from deep_learning_template.registry import METRIC_FUNCTIONS
from .accuracy import accuracy


@METRIC_FUNCTIONS.register('accuracy')
def build_accuracy(cfg):
    return accuracy


def build_metric_functions(cfg):
    metric_functions = {}
    for k in cfg.solver.metric_functions:
        v = METRIC_FUNCTIONS[k](cfg)
        metric_functions[k] = v
    return metric_functions
