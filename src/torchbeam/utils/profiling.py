from time import perf_counter
from torch.cuda import synchronize
from collections import deque

__all__ = ["AverageTimer"]


class AverageTimer:

    _timers = {}
    _nesting_level = 0

    def __init__(self, task_name, cuda_sync=False, n=20):
        self.task_name = task_name
        self.cuda_sync = cuda_sync
        self.n = n

    def __enter__(self):
        AverageTimer._nesting_level += 1
        if self.cuda_sync:
            synchronize()
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cuda_sync:
            synchronize()
        self.end_time = perf_counter()
        elapsed_time = self.end_time - self.start_time
        if self.task_name not in AverageTimer._timers:
            AverageTimer._timers[self.task_name] = deque([], self.n)
        time_list = AverageTimer._timers[self.task_name]
        time_list.append(elapsed_time)
        indentation = "\t" * (AverageTimer._nesting_level - 1)
        average = sum(time_list) / len(time_list)
        print(f"{indentation}{self.task_name}: {1e3 * average:.3f} ms")
        AverageTimer._nesting_level -= 1
