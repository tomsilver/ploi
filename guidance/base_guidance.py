"""Base class for learning search guidance for a planner by scoring objects.
"""

import abc


class BaseSearchGuidance:
    """Abstract class for search guidance that suggests which objects to
    use when planning.
    """
    @abc.abstractmethod
    def train(self, train_env_name):
        """Train whatever is needed for scoring at test time.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def seed(self, seed):
        """Seed this search guidance method.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def score_object(self, obj, state):
        """Return a score for the given object, from 0-1.
        """
        raise NotImplementedError("Override me!")
