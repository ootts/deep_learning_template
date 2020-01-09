from deep_learning_template.registry import BATCH_COLLATORS
from .default_batch_collator import DefaultBatchCollator


@BATCH_COLLATORS.register('DefaultBatchCollator')
def build_default_batch_colloator(cfg):
    return DefaultBatchCollator()


def make_batch_collator(cfg):
    return BATCH_COLLATORS[cfg.dataloader.collator](cfg)
