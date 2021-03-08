import os.path as osp
import os

import torch

from deep_learning_template.config import cfg
from deep_learning_template.engine.defaults import default_argument_parser, default_setup
from deep_learning_template.engine.launch import launch
from deep_learning_template.evaluators.build import build_evaluators
from deep_learning_template.trainer.build import build_trainer
from deep_learning_template.utils.comm import get_world_size, get_rank
from deep_learning_template.utils.os_utils import isckpt
from deep_learning_template.visualizers.build import build_visualizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.output_dir == '':
        assert args.config_file.startswith('configs') and args.config_file.endswith('.yaml')
        cfg.output_dir = args.config_file[:-5].replace('configs', 'models')
    cfg.freeze()
    os.makedirs(cfg.output_dir, exist_ok=True)
    default_setup(cfg, args)
    return cfg


def main():
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main_func,
        args.num_gpus,
        dist_url=args.dist_url,
        args=(args,),
    )


def eval_one_ckpt(trainer):
    preds = trainer.get_preds()
    if get_rank() == 0:
        if not cfg.test.skip_evaluation:
            evaluators = build_evaluators(cfg)
            for evaluator in evaluators:
                evaluator(preds, trainer.valid_dl.dataset)
        if not cfg.test.skip_visualization:
            visualizer = build_visualizer(cfg)
            visualizer(preds, trainer.valid_dl.dataset)


def eval_all_ckpts(trainer):
    if not cfg.test.skip_evaluation:
        evaluators = build_evaluators(cfg)
    if not cfg.test.skip_visualization:
        visualizer = build_visualizer(cfg)
    for fname in sorted(os.listdir(cfg.output_dir)):
        if isckpt(fname):
            cfg.defrost()
            cfg.solver.load = fname[:-4]
            cfg.freeze()
            trainer.resume()
            preds = trainer.get_preds()
            if not cfg.test.skip_evaluation:
                for evaluator in evaluators:
                    evaluator(preds, trainer.valid_dl.dataset)
            if not cfg.test.skip_visualization:
                visualizer(preds, trainer.valid_dl.dataset)


def main_func(args):
    world_size = get_world_size()
    distributed = world_size > 1
    cfg = setup(args)
    trainer = build_trainer(cfg)
    trainer.resume()
    if distributed:
        trainer.to_distributed()
    if cfg.test.eval_all:
        eval_all_ckpts(trainer)
    else:
        eval_one_ckpt(trainer)


if __name__ == "__main__":
    main()
