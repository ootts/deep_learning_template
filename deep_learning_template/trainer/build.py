from deep_learning_template.registry import TRAINERS
from deep_learning_template.trainer.base import BaseTrainer
from .mnist import MnistTrainer


@TRAINERS.register('base')
def build_base_trainer(*args):
    return BaseTrainer(*args)


@TRAINERS.register('mnist')
def build_mnist_trainer(*args):
    return MnistTrainer(*args)


def build_trainer(cfg, model, train_dl, valid_dl, epochs,
                  loss_function, optimizer, scheduler, output_dir, max_lr,
                  save_every, metric_functions, logger) -> BaseTrainer:
    return TRAINERS[cfg.solver.trainer](model, train_dl, valid_dl, epochs,
                                        loss_function, optimizer, scheduler, output_dir, max_lr,
                                        save_every, metric_functions, logger)
