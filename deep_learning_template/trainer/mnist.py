import os
import time

from dl_ext.average_meter import AverageMeter
from dl_ext.pytorch_ext.dist import *
from termcolor import colored
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_learning_template.trainer.base import BaseTrainer
from .utils import *


class MnistTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, train_dl: DataLoader, valid_dl: DataLoader, num_epochs: int,
                 loss_function: callable, optimizer: Optimizer, scheduler: _LRScheduler = None,
                 output_dir: str = 'models', max_lr: float = 1e-2, save_every: bool = False,
                 metric_functions: dict = None, logger=None):

        super().__init__(model, train_dl, valid_dl, num_epochs, loss_function, optimizer, scheduler, output_dir, max_lr,
                         save_every, metric_functions, logger)
        self.best_val_acc = 0.0

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
            return metric_ams['accuracy'].avg

    def fit(self):
        os.makedirs(self.output_dir, exist_ok=True)
        num_epochs = self.num_epochs
        begin = time.time()
        for epoch in range(self.begin_epoch, num_epochs):
            self.train(epoch)
            synchronize()
            val_acc = self.val(epoch)
            synchronize()
            if is_main_process():
                if self.save_every:
                    self.save(epoch)
                elif val_acc > self.best_val_acc:
                    self.logger.info(
                        colored('Better model found at epoch %d with val_acc %.4f.' % (epoch, val_acc), 'red'))
                    self.best_val_acc = val_acc
                    self.save(epoch)
            synchronize()
        if is_main_process():
            self.logger.info('Training finished. Total time %s' % (format_time(time.time() - begin)))

    def save(self, epoch):
        name = os.path.join(self.output_dir, str(epoch) + '.pth')
        net_sd = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        d = {'model': net_sd,
             'optimizer': self.optimizer.state_dict(),
             'scheduler': self.scheduler.state_dict(),
             'epoch': epoch,
             'best_val_acc': self.best_val_acc}
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
        self.best_val_acc = d['best_val_acc']
