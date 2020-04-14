import os
import argparse
import sys

from deep_learning_template.utils.comm import get_rank, get_world_size
from deep_learning_template.utils.logger import setup_logger
from deep_learning_template.utils.miscellaneous import save_config


def default_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--mode", default="train", choices=['train', 'eval', 'findlr'])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")

    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def default_setup(cfg, args):
    logger = setup_logger("deep_learning_template", cfg.output_dir, get_rank())
    world_size = get_world_size()
    logger.info("Using {} GPUs".format(world_size))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("\n" + open(args.config_file, "r").read())
    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(cfg.output_dir, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)
