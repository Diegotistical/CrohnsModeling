"""
Task scheduler for simulation loop.
"""
from typing import Callable, List, Dict, Any

class Scheduler:
    def __init__(self):
        self.tasks: List[Dict[str, Any]] = []

    def every(self, steps: int, func: Callable, *args, **kwargs):
        """Register a callback to run every N steps."""
        self.tasks.append({
            'interval': steps,
            'func': func,
            'args': args,
            'kwargs': kwargs
        })

    def __call__(self, step_i: int, *args, **kwargs):
        """Executor called by the solver loop."""
        for task in self.tasks:
            if step_i % task['interval'] == 0:
                task['func'](*task['args'], **task['kwargs'])