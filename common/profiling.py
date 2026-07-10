import time

class Profiler:
    def __init__(self):
        self._time_in = None
        self._current_category = 'uncategorised'
        self._times = {}
        self.count = 0
    
    def time_in(self, category = 'uncategorised'):
        assert self._time_in is None, 'attempted to start timing when already timing'
        self._time_in = time.time()
        self._current_category = category
    
    def time_out(self):
        assert self._time_in is not None, 'attempted to stop timing without having started it'
        elapsed = time.time() - self._time_in
        if self._current_category not in self._times:
            self._times[self._current_category] = 0
        self._times[self._current_category] += elapsed
        self._time_in = None
        self._current_category = 'uncategorised'

        return elapsed
    
    def elapsed_in(self, category):
        if category not in self._times:
            return 0
        return self._times[category]
    
    def times(self):
        return self._times
    
    def count_up(self):
        self.count += 1
