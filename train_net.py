import argparse
import os

import torch
import torch.multiprocessing
from dl_ext.pytorch_ext import synchronize, get_rank
from torch.distributed import init_process_group
from deep_learning_template.config import cfg
from deep_learning_template.data import make_data_loader
from deep_learning_template.evaluators.build import build_evaluator
from deep_learning_template.modeling.models import build_model
from deep_learning_template.solver.build import make_optimizer, make_lr_scheduler
from deep_learning_template.trainer.build import build_trainer
from deep_learning_template.utils.logger import setup_logger
from deep_learning_template.utils.miscellaneous import save_config
from deep_learning_template.loss.build import build_loss_function
from deep_learning_template.metric.build import build_metric_functions

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="configs/default.yaml",
        metavar="FILE",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--mode", type=str, default='train',
                        choices=['train', 'eval', 'findlr'])
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.output_dir == '':
        assert args.config_file.startswith('configs') and args.config_file.endswith('.yaml')
        cfg.output_dir = args.config_file[:-5].replace('configs', 'models')
    cfg.freeze()
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = setup_logger("deep_learning_template", cfg.output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("\n" + open(args.config_file, "r").read())
    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(cfg.output_dir, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)
    model = build_model(cfg).to(torch.device(cfg.model.device))

    output_dir = cfg.output_dir
    train_dl = make_data_loader(cfg, is_train=True)
    valid_dl = make_data_loader(cfg, is_train=False)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer, cfg.solver.epochs * len(train_dl))
    loss_function = build_loss_function(cfg)
    metric_functions = build_metric_functions(cfg)
    trainer = build_trainer(cfg, model, train_dl, valid_dl, cfg.solver.epochs,
                            loss_function, optimizer, scheduler, output_dir, cfg.solver.max_lr,
                            cfg.solver.save_every, metric_functions, logger)
    if cfg.model.load_model != '':
        logger.info('loading model from %s' % cfg.model.load_model)
        trainer.load_model(cfg.model.load_model)
    if cfg.model.load != '':
        logger.info('loading checkpoint from %s' % cfg.model.load)
        trainer.load(cfg.model.load)

    evaluator = build_evaluator(cfg)
    if distributed:
        trainer.to_distributed()
    if args.mode == 'train':
        trainer.fit()
    elif args.mode == 'eval':
        preds = trainer.get_preds()
        evaluator(preds, valid_dl.dataset)
    else:
        trainer.to_base()
        trainer.find_lr()


if __name__ == "__main__":
    main()
