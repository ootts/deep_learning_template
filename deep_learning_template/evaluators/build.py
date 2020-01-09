import torch
from tqdm import tqdm

from deep_learning_template.metric.accuracy import accuracy
from deep_learning_template.registry import EVALUATORS


@EVALUATORS.register('mnist')
def mnist_evaluator(cfg):
    class E:
        def __call__(self, x, ds):
            y = []
            print('collecting targets...')
            for batch in tqdm(ds):
                y.append(batch[1])
            y = torch.tensor(y).to(x.device)
            acc = accuracy(x, y)
            print(acc.item())

    return E()


def build_evaluator(cfg):
    return EVALUATORS[cfg.test.evaluator](cfg)
