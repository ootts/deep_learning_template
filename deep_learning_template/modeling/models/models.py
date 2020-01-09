from .lenet5 import LeNet5

_META_ARCHITECTURES = {'LeNet5': LeNet5}


def build_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    return meta_arch(cfg)
