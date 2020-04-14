import torch
from tqdm import tqdm

from deep_learning_template.metric.accuracy import accuracy
from deep_learning_template.registry import EVALUATORS
from deep_learning_template.utils import comm


class DefaultEvaluator:
    def __init__(self, cfg):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class MnistEvaluator(DefaultEvaluator):
    def __call__(self, x, ds):
        if comm.get_rank() == 0:
            y = []
            print('collecting targets...')
            for batch in tqdm(ds):
                y.append(batch[1])
            y = torch.tensor(y).to(x.device)
            acc = accuracy(x, y)
            print(acc.item())


CIFAR10Evaluator = MnistEvaluator


@EVALUATORS.register('mnist')
def mnist_evaluator(cfg):
    return MnistEvaluator(cfg)


@EVALUATORS.register('cifar10')
def cifar10_evaluator(cfg):
    return CIFAR10Evaluator(cfg)


def build_evaluator(cfg):
    return EVALUATORS[cfg.test.evaluator](cfg)
