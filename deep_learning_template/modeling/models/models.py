from .lenet5 import LeNet5
from .resnet import ResNet18

_META_ARCHITECTURES = {'LeNet5': LeNet5, 'ResNet18': ResNet18}


def build_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    return meta_arch(cfg)
