import os
from yacs.config import CfgNode as CN

_C = CN()

_C.model = CN()
_C.model.device = "cuda"
_C.model.meta_architecture = "GeneralizedRCNN"

_C.model.load_model = ""
_C.model.load = ""

_C.input = CN()
_C.input.min_size_train = (600,)
_C.input.max_size_train = 2000
_C.input.min_size_test = 600
_C.input.max_size_test = 2000
_C.input.do_normalize = True
_C.input.pixel_mean = [0.485, 0.456, 0.406]
_C.input.pixel_std = [0.229, 0.224, 0.225]
_C.input.horizontal_flip_prob = 0.5
_C.input.shuffle = True
_C.input.brightness = 0.0
_C.input.contrast = 0.0
_C.input.saturation = 0.0
_C.input.hue = 0.0

_C.datasets = CN()
_C.datasets.train = ()
_C.datasets.test = ()

_C.dataloader = CN()
_C.dataloader.num_workers = 0
_C.dataloader.collator = 'DefaultBatchCollator'

_C.solver = CN()
_C.solver.epochs = 1
_C.solver.max_lr = 0.01
_C.solver.bias_lr_factor = 2
_C.solver.momentum = 0.9
_C.solver.weight_decay = 0.0005
_C.solver.weight_decay_bias = 0
_C.solver.gamma = 0.1
_C.solver.steps = (30000,)
_C.solver.warmup_factor = 1.0 / 3
_C.solver.warmup_iters = 500
_C.solver.warmup_method = "linear"
_C.solver.optimizer = 'Adam'
_C.solver.scheduler = 'OneCycleScheduler'
_C.solver.do_grad_clip = True
_C.solver.grad_clip = 1.0
_C.solver.ds_len = -1
_C.solver.batch_size = 2
_C.solver.loss_function = ''
_C.solver.save_every = False
_C.solver.metric_functions = ('',)
_C.solver.trainer = "base"

_C.test = CN()
_C.test.batch_size = 2
_C.test.evaluator = ''

_C.output_dir = ''

_C.paths_catalog = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
