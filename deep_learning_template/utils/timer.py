import time
import datetime

import torch


class Timer(object):
    def __init__(self):
        self.reset()

    @property
    def average_time(self):
        return self.total_time / self.calls if self.calls > 0 else 0.0

    def tic(self, sync_cuda=False):
        if sync_cuda:
            torch.cuda.synchronize()
        self.start_time = time.time()

    def toc(self, sync_cuda=False):
        if sync_cuda:
            torch.cuda.synchronize()
        self.add(time.time() - self.start_time)

    def add(self, time_diff):
        self.diff = time_diff
        self.total_time += self.diff
        self.calls += 1

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0

    def avg_time_str(self):
        time_str = str(datetime.timedelta(seconds=self.average_time))
        return time_str
