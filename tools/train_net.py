import os

from deep_learning_template.config import cfg
from deep_learning_template.engine.defaults import default_argument_parser, default_setup
from deep_learning_template.engine.launch import launch
from deep_learning_template.trainer.build import build_trainer
from deep_learning_template.utils.comm import get_world_size


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


def main_func(args):
    world_size = get_world_size()
    distributed = world_size > 1
    cfg = setup(args)
    trainer = build_trainer(cfg)
    if args.resume:
        trainer.resume()
    if distributed:
        trainer.to_distributed()
    if args.mode == 'train':
        trainer.fit()
    elif args.mode == 'findlr':
        trainer.to_base()
        trainer.find_lr()
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
