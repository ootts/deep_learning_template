from torch import Tensor


def accuracy(x: Tensor, y: Tensor):
    return (x.argmax(-1) == y).sum().float() / y.shape[0]
