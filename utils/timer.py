import functools
import numpy as np
import time
import torch

"""
Usage:
    from qr_utils import timer

    # this may be in a loop
    with timer.time_as("loading images"):
        /* code that loads images */

    # this may be in a loop
    with timer.time_as("computation"):
        /* code that does computation */

    @timer.time_fn("my_function")
    def my_function():
        /* function implementation */

    timer.print_stats()
"""

class Timer:
    def __init__(self, num_skip=0, pytorch=False):
        self.ttime = {}  # total time for every key
        self.ttimesq = {}  # total time-squared for every key
        self.titer = {}  # total number of iterations for every key

        self.pytorch = pytorch
        self.num_skip = num_skip
        self.tskip = {}  # total skips remaining for every key

    def clear(self):
        self.ttime.clear()
        self.ttimesq.clear()
        self.titer.clear()
        self.tskip.clear()

    def _add(self, key, time_):
        if key not in self.ttime:
            self.ttime[key] = 0
            self.ttimesq[key] = 0
            self.titer[key] = 0
            self.tskip[key] = self.num_skip

        if self.tskip[key] > 0:
            self.tskip[key] -= 1
        else:
            self.ttime[key] += time_
            self.ttimesq[key] += time_ * time_
            self.titer[key] += 1

    def get_stats_dict(self):
        stats = {}
        for key in self.ttime:
            ttime_, ttimesq_, titer_ = self.ttime[key], self.ttimesq[key], self.titer[key]
            if titer_ == 0: continue
            mean = ttime_ / titer_
            std = np.sqrt(ttimesq_ / titer_ - mean * mean)
            interval = 1.96 * std
            stats[key] = dict(mean=mean, std=std, interval=interval)
        return stats

    def get_stats_string(self):
        ret = "TIMER STATS:\n"
        stats = self.get_stats_dict()
        if len(stats) == 0: return
        word_len = max([len(k) for k in stats.keys()]) + 8
        for key, stat in stats.items():
            mean, interval = stat["mean"], stat["interval"]
            ret += f"{key.rjust(word_len)}: {mean*1e3:.1f}ms ± {interval*1e3:.1f}ms\n"
        return ret

    def print_stats(self):
        print(self.get_stats_string())

    class TimerContext:
        def __init__(self, timer: 'Timer', key):
            self.timer = timer
            self.key = key

            self._stime = 0
            self._etime = 0

        def __enter__(self):
            if self.timer.pytorch: torch.cuda.synchronize()
            self._stime = time.time()

        def __exit__(self, type, value, traceback):
            if self.timer.pytorch: torch.cuda.synchronize()
            self._etime = time.time()
            time_ = self._etime - self._stime
            self.timer._add(self.key, time_)

    def time_as(self, key):
        return Timer.TimerContext(self, key)

    def time_fn(self, name):
        def decorator(function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                with self.time_as(name):
                    ret = function(*args, **kwargs)
                # Previously, we would print every time the function was called
                # ttime_   = self.ttime[name]
                # titer_   = self.titer[name]
                # avg_time = ttime_ / titer_
                # print(f"Avg. {name} time is {datetime.timedelta(seconds=round(avg_time))}s")
                return ret
            return wrapper
        return decorator


timer = Timer(num_skip=0)
