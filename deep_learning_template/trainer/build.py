from deep_learning_template.registry import TRAINERS
from deep_learning_template.trainer.base import BaseTrainer
from .mnist import MnistTrainer
from .cifar10 import CIFAR10Trainer


@TRAINERS.register('base')
def build_base_trainer(cfg):
    return BaseTrainer(cfg)


@TRAINERS.register('mnist')
def build_mnist_trainer(cfg):
    return MnistTrainer(cfg)


@TRAINERS.register('cifar10')
def build_cifar10_trainer(cfg):
    return CIFAR10Trainer(cfg)


def build_trainer(cfg) -> BaseTrainer:
    return TRAINERS[cfg.solver.trainer](cfg)
