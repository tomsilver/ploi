"""General interface for a planner.
"""

import abc


class Planner:
    """An abstract planner for PDDLGym.
    """
    @abc.abstractmethod
    def __call__(self, domain, state, timeout):
        """Takes in a PDDLGym domain and PDDLGym state. Returns a plan.
        """
        raise NotImplementedError("Override me!")


class PlanningFailure(Exception):
    """Exception raised when planning fails.
    """
    pass


class PlanningTimeout(Exception):
    """Exception raised when planning times out.
    """
    pass
