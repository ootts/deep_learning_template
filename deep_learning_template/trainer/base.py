import os.path as osp
import logging
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from dl_ext.average_meter import AverageMeter
from dl_ext.pytorch_ext.dist import *
from dl_ext.pytorch_ext.optim import OneCycleScheduler, LRFinder

from matplotlib import axes, figure
from tensorboardX import SummaryWriter
from termcolor import colored
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from deep_learning_template.data.samplers.ordered_distributed_sampler import OrderedDistributedSampler
from .utils import *
from ..data import make_data_loader
from ..loss.build import build_loss_function
from ..metric.build import build_metric_functions
from ..modeling.models import build_model
from ..solver.build import make_optimizer, make_lr_scheduler
from ..utils.tb_utils import get_summary_writer


class BaseTrainer:

    def __init__(self, cfg):
        self.model: nn.Module = build_model(cfg).to(torch.device(cfg.model.device))
        self.loss_function = build_loss_function(cfg)
        self.train_dl = make_data_loader(cfg, is_train=True)
        self.valid_dl = make_data_loader(cfg, is_train=False)
        self.output_dir = cfg.output_dir
        self.num_epochs = cfg.solver.num_epochs
        self.begin_epoch = 0
        self.max_lr = cfg.solver.max_lr
        self.skip_validation = cfg.solver.skip_validation
        self.save_every = cfg.solver.save_every
        if self.skip_validation and not self.save_every:
            raise RuntimeError('model must be save every [epoch/iteration] when validation is skipped.')
        self.save_mode = cfg.solver.save_mode
        self.save_freq = cfg.solver.save_freq
        self.optimizer = make_optimizer(cfg, self.model)
        self.scheduler = make_lr_scheduler(cfg, self.optimizer,
                                           cfg.solver.num_epochs * len(self.train_dl))
        self.metric_functions = build_metric_functions(cfg)
        self.cfg = cfg
        self._tb_writer = None
        self.state = TrainerState.BASE
        self.global_steps = 0
        self.best_val_loss = 100000
        self.val_loss = 100000
        self.logger = self._setup_logger()

    def train(self, epoch):
        loss_meter = AverageMeter()
        metric_ams = {}
        for metric in self.metric_functions.keys():
            metric_ams[metric] = AverageMeter()
        self.model.train()
        bar = tqdm(self.train_dl, leave=False) if is_main_process() else self.train_dl
        begin = time.time()
        for batch in bar:
            self.optimizer.zero_grad()
            x, y = batch_gpu(batch)
            output = self.model(x)
            loss = self.loss_function(output, y)
            loss = loss.mean()
            loss.backward()
            if self.cfg.solver.do_grad_clip:
                if self.cfg.solver.grad_clip_type == 'norm':
                    clip_grad_norm_(self.model.parameters(), self.cfg.solver.grad_clip)
                else:
                    clip_grad_value_(self.model.parameters(), self.cfg.solver.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None and isinstance(self.scheduler, OneCycleScheduler):
                self.scheduler.step()
            # record and plot loss and metrics
            reduced_loss = reduce_loss(loss)
            metrics = {}
            for metric, f in self.metric_functions.items():
                s = f(output, y).mean()
                reduced_s = reduce_loss(s)
                metrics[metric] = reduced_s
            if is_main_process():
                loss_meter.update(reduced_loss.item())
                lr = self.optimizer.param_groups[0]['lr']
                self.tb_writer.add_scalar('train/loss', reduced_loss.item(), self.global_steps)
                self.tb_writer.add_scalar('train/lr', lr, self.global_steps)
                bar_vals = {'epoch': epoch, 'phase': 'train', 'loss': loss_meter.avg, 'lr': lr}
                for k, v in metrics.items():
                    metric_ams[k].update(v.item())
                    self.tb_writer.add_scalar(f'train/{k}', v.item(), self.global_steps)
                    bar_vals[k] = metric_ams[k].avg
                bar.set_postfix(bar_vals)
            self.global_steps += 1
            if self.global_steps % self.save_freq == 0:
                self.try_to_save(epoch, 'iteration')
        torch.cuda.synchronize()
        epoch_time = format_time(time.time() - begin)
        if is_main_process():
            metric_msgs = ['epoch %d, train, loss %.4f, time %s' % (
                epoch, loss_meter.avg, epoch_time)]
            for metric, v in metric_ams.items():
                metric_msgs.append('%s %.4f' % (metric, v.avg))
            s = ', '.join(metric_msgs)
            self.logger.info(s)
        if self.scheduler is not None and not isinstance(self.scheduler, OneCycleScheduler):
            self.scheduler.step()

    @torch.no_grad()
    def val(self, epoch):
        loss_meter = AverageMeter()
        metric_ams = {}
        for metric in self.metric_functions.keys():
            metric_ams[metric] = AverageMeter()
        self.model.eval()
        bar = tqdm(self.valid_dl, leave=False) if is_main_process() else self.valid_dl
        begin = time.time()
        for batch in bar:
            x, y = batch_gpu(batch)
            output = self.model(x)
            loss = self.loss_function(output, y)
            loss = loss.mean()
            reduced_loss = reduce_loss(loss)
            metrics = {}
            for metric, f in self.metric_functions.items():
                s = f(output, y).mean()
                reduced_s = reduce_loss(s)
                metrics[metric] = reduced_s
            if is_main_process():
                loss_meter.update(reduced_loss.item())
                bar_vals = {'epoch': epoch, 'phase': 'val', 'loss': loss_meter.avg}
                for k, v in metrics.items():
                    metric_ams[k].update(v.item())
                    bar_vals[k] = metric_ams[k].avg
                bar.set_postfix(bar_vals)
        torch.cuda.synchronize()
        epoch_time = format_time(time.time() - begin)
        if is_main_process():
            metric_msgs = ['epoch %d, val, loss %.4f, time %s' % (
                epoch, loss_meter.avg, epoch_time)]
            for metric, v in metric_ams.items():
                metric_msgs.append('%s %.4f' % (metric, v.avg))
            s = ', '.join(metric_msgs)
            self.logger.info(s)
            self.tb_writer.add_scalar('val/loss', loss_meter.avg, epoch)
            for metric, s in metric_ams.items():
                self.tb_writer.add_scalar(f'val/{metric}', s.avg, epoch)
            return loss_meter.avg

    def fit(self):
        os.makedirs(self.output_dir, exist_ok=True)
        num_epochs = self.num_epochs
        begin = time.time()
        for epoch in range(self.begin_epoch, num_epochs):
            self.train(epoch)
            synchronize()
            if not self.skip_validation:
                self.val_loss = self.val(epoch)
                synchronize()
            self.try_to_save(epoch, 'epoch')
            # if is_main_process():
            #     if self.save_every and (
            #             self.save_mode == 'epoch' and epoch % self.cfg.solver.save_freq == 0 or epoch == num_epochs - 1):
            #         self.save(epoch)
            #     elif val_loss < self.best_val_loss:
            #         self.logger.info(
            #             colored('Better model found at epoch %d with val_loss %.4f.' % (epoch, val_loss), 'red'))
            #         self.best_val_loss = val_loss
            #         self.save(epoch)
            synchronize()
        if is_main_process():
            self.logger.info('Training finished. Total time %s' % (format_time(time.time() - begin)))

    @torch.no_grad()
    def get_preds(self):
        prediction_path = osp.join(self.cfg.output_dir, 'inference', self.cfg.datasets.test, 'predictions.pth')
        if not self.cfg.test.force_recompute and osp.exists(prediction_path):
            self.logger.info(colored(f'predictions found at {prediction_path}, skip recomputing.', 'red'))
            outputs = torch.load(prediction_path)
        else:
            if get_world_size() > 1:
                outputs = self.get_preds_dist()
            else:
                self.model.eval()
                ordered_valid_dl = DataLoader(self.valid_dl.dataset, self.valid_dl.batch_size, shuffle=False,
                                              sampler=None, num_workers=self.valid_dl.num_workers,
                                              collate_fn=self.valid_dl.collate_fn, pin_memory=self.valid_dl.pin_memory,
                                              timeout=self.valid_dl.timeout,
                                              worker_init_fn=self.valid_dl.worker_init_fn)
                bar = tqdm(ordered_valid_dl)
                outputs = []
                for batch in bar:
                    x, y = batch_gpu(batch)
                    output = self.model(x)
                    output = to_cpu(output)
                    outputs.append(output)
                outputs = torch.cat(outputs)
            os.makedirs(osp.dirname(prediction_path), exist_ok=True)
            torch.save(outputs, prediction_path)
        return outputs

    @torch.no_grad()
    def get_preds_dist(self):
        self.model.eval()
        valid_sampler = OrderedDistributedSampler(self.valid_dl.dataset, get_world_size(), rank=get_rank())
        ordered_dist_valid_dl = DataLoader(self.valid_dl.dataset, self.valid_dl.batch_size, shuffle=False,
                                           sampler=valid_sampler, num_workers=self.valid_dl.num_workers,
                                           collate_fn=self.valid_dl.collate_fn, pin_memory=self.valid_dl.pin_memory,
                                           timeout=self.valid_dl.timeout,
                                           worker_init_fn=self.valid_dl.worker_init_fn)
        bar = tqdm(ordered_dist_valid_dl) if is_main_process() else ordered_dist_valid_dl
        outputs = []
        for batch in bar:
            x, y = batch_gpu(batch)
            output = self.model(x)
            output = to_cpu(output)
            outputs.append(output)
        outputs = torch.cat(outputs)
        all_outputs = all_gather(outputs)
        if not is_main_process():
            return
        all_outputs = torch.cat(all_outputs, dim=0).cpu()[:len(self.valid_dl.dataset)]
        return all_outputs

    def to_base(self):
        if self.state == TrainerState.BASE:
            return
        elif self.state == TrainerState.PARALLEL:
            self.model = self.model.module
            if isinstance(self.scheduler, OneCycleScheduler):
                world_size = get_world_size()
                self.scheduler.total_steps *= world_size
                self.scheduler.step_size_up *= world_size
                self.scheduler.step_size_down *= world_size
        else:
            self.model = self.model.module
            self.train_dl = self.old_train_dl
            self.valid_dl = self.old_valid_dl
            if isinstance(self.scheduler, OneCycleScheduler):
                world_size = get_world_size()
                self.scheduler.total_steps *= world_size
                self.scheduler.step_size_up *= world_size
                self.scheduler.step_size_down *= world_size

    def to_parallel(self):
        assert self.state == TrainerState.BASE
        devices = os.environ['CUDA_VISIBLE_DEVICES']
        print('visible devices', devices)
        self.model = DataParallel(self.model)
        if isinstance(self.scheduler, OneCycleScheduler):
            world_size = get_world_size()
            self.scheduler.total_steps //= world_size
            self.scheduler.step_size_up //= world_size
            self.scheduler.step_size_down //= world_size

    def to_distributed(self, convert_sync_batchnorm=True):
        assert dist.is_available() and dist.is_initialized()
        local_rank = dist.get_rank()
        if convert_sync_batchnorm:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DistributedDataParallel(self.model, [local_rank],
                                             output_device=local_rank,
                                             broadcast_buffers=False)
        self.old_train_dl = self.train_dl
        train_sampler = DistributedSampler(self.train_dl.dataset, shuffle=True)
        new_train_dl = DataLoader(self.train_dl.dataset, self.train_dl.batch_size, shuffle=False,
                                  sampler=train_sampler, num_workers=self.train_dl.num_workers,
                                  collate_fn=self.train_dl.collate_fn, pin_memory=self.train_dl.pin_memory,
                                  timeout=self.train_dl.timeout, worker_init_fn=self.train_dl.worker_init_fn)
        self.train_dl = new_train_dl
        self.old_valid_dl = self.valid_dl
        valid_sampler = DistributedSampler(self.valid_dl.dataset, shuffle=False)
        new_valid_dl = DataLoader(self.valid_dl.dataset, self.valid_dl.batch_size, shuffle=False,
                                  sampler=valid_sampler, num_workers=self.valid_dl.num_workers,
                                  collate_fn=self.valid_dl.collate_fn, pin_memory=self.valid_dl.pin_memory,
                                  timeout=self.valid_dl.timeout, worker_init_fn=self.valid_dl.worker_init_fn)
        self.valid_dl = new_valid_dl
        if isinstance(self.scheduler, OneCycleScheduler):
            world_size = get_world_size()
            self.scheduler.total_steps /= world_size
            self.scheduler.step_size_up /= world_size
            self.scheduler.step_size_down /= world_size

    def find_lr(self, start_lr: float = 1e-7, end_lr: float = 10,
                num_it: int = 100, stop_div: bool = True,
                skip_start: int = 10, skip_end: int = 5, suggestion: bool = True):
        assert self.state == TrainerState.BASE
        self.old_scheduler = self.scheduler
        self.scheduler = LRFinder(self.optimizer, start_lr, end_lr, num_it, stop_div)
        loss_meter = AverageMeter()
        self.model.train()

        it = 0
        lrs, smooth_losses = [], []
        best_loss = 10000
        for epoch in range(round(math.ceil(num_it / len(self.train_dl)))):
            bar = tqdm(self.train_dl, leave=False)
            for batch in bar:
                if it > num_it: break
                self.optimizer.zero_grad()
                x, y = batch_gpu(batch)
                output = self.model(x)
                loss = self.loss_function(output, y)
                loss = loss.mean()
                if (loss > 40 * best_loss or torch.isnan(loss).sum() != 0) and stop_div:
                    print('loss diverge, stop.')
                    break
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                # record and plot loss and metrics
                loss_meter.update(loss.item())
                best_loss = min(loss.item(), best_loss)
                lr = self.optimizer.param_groups[0]['lr']
                lrs.append(lr)
                smooth_losses.append(loss_meter.avg)
                bar_vals = {'it': it, 'phase': 'train', 'loss': loss_meter.avg, 'lr': lr}
                bar.set_postfix(bar_vals)
                it += 1
        lrs = split_list(lrs, skip_start, skip_end)
        losses = split_list(smooth_losses, skip_start, skip_end)
        # losses = [x() for x in losses]
        fig, ax = plt.subplots(1, 1)
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        if suggestion:
            try:
                mg = (np.gradient(np.array(losses))).argmin()
            except:
                print("Failed to compute the gradients, there might not be enough points.")
                return
            print(f"Min numerical gradient: {lrs[mg]:.2E}")
            ax.plot(lrs[mg], losses[mg], markersize=10, marker='o', color='red')
            ml = np.argmin(losses)
            print(f"Min loss divided by 10: {lrs[ml] / 10:.2E}")
        fig: figure.Figure
        ax: axes.Axes
        fig.savefig(os.path.join(self.output_dir, 'lr.jpg'))
        # reset scheduler
        self.scheduler = self.old_scheduler

    def save(self, epoch):
        if self.save_mode == 'epoch':
            name = os.path.join(self.output_dir, 'model_epoch_%06d.pth' % epoch)
        else:
            name = os.path.join(self.output_dir, 'model_iteration_%06d.pth' % self.global_steps)
        net_sd = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        d = {'model': net_sd,
             'optimizer': self.optimizer.state_dict(),
             'scheduler': self.scheduler.state_dict(),
             'epoch': epoch,
             'best_val_loss': self.best_val_loss,
             'global_steps': self.global_steps}
        torch.save(d, name)

    def load(self, name):
        name = os.path.join(self.output_dir, name + '.pth')
        d = torch.load(name, 'cpu')
        net_sd = d['model']
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(net_sd)
        else:
            self.model.load_state_dict(net_sd)
        self.optimizer.load_state_dict(d['optimizer'])
        self.scheduler.load_state_dict(d['scheduler'])
        self.begin_epoch = d['epoch']
        self.best_val_loss = d['best_val_loss']
        if 'global_steps' in d:  # compat
            self.global_steps = d['global_steps']

    def load_model(self, name):
        d = torch.load(name, 'cpu')
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(d)
        else:
            self.model.load_state_dict(d)

    def _setup_logger(self):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        # don't log results for the non-master process
        if get_rank() > 0:
            return logger
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(self.output_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    @property
    def tb_writer(self):
        if self._tb_writer is None and is_main_process():
            self._tb_writer = get_summary_writer(self.output_dir, flush_secs=20)
        return self._tb_writer

    def resume(self):
        if self.cfg.solver.load == '' and self.cfg.solver.load_model == '':
            self.logger.warning('try to resume without loading anything!')
        if self.cfg.solver.load_model != '':
            self.logger.info(colored('loading model from %s' % self.cfg.solver.load_model, 'red'))
            self.load_model(self.cfg.solver.load_model)
        if self.cfg.solver.load != '':
            self.logger.info(colored('loading checkpoint from %s' % self.cfg.solver.load, 'red'))
            self.load(self.cfg.solver.load)

    def try_to_save(self, epoch, flag):
        if not is_main_process():
            return
        if self.skip_validation:
            # validation is not performed, must save every epoch/iteration
            assert self.save_every
            if flag == self.save_mode:
                self.save(epoch)
        else:
            # validation is performed, can save according to val_loss or every epoch
            if self.save_every:
                # save every x epochs
                assert self.save_mode == 'epoch'
                if epoch % self.save_freq == 0 or epoch == self.num_epochs - 1:
                    self.save(epoch)
            if self.val_loss < self.best_val_loss:
                self.logger.info(
                    colored('Better model found at epoch'
                            ' %d with val_loss %.4f.' % (epoch, self.val_loss), 'red'))
                self.save(epoch)
