from deep_learning_template.registry import VISUALIZERS


@VISUALIZERS.register('default')
class DefaultVisualizer:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, *args, **kwargs):
        return


def build_visualizer(cfg):
    return VISUALIZERS[cfg.test.visualizer](cfg)
