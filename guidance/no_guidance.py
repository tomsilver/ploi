"""Uniformly random search guidance.
"""

import numpy as np
from PLOI.guidance import BaseSearchGuidance


class NoSearchGuidance(BaseSearchGuidance):
    """Uniform/no search guidance.
    """
    def __init__(self):
        super().__init__()
        self._rng = None

    def train(self, train_env_name):
        pass  # unused

    def seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def score_object(self, obj, state):
        return self._rng.uniform()
