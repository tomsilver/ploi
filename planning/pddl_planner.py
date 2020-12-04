"""General interface for a PDDL planner.
"""

import os
import time
import abc
import tempfile
import subprocess
from pddlgym.spaces import LiteralSpace
from pddlgym.parser import parse_plan_step, PDDLProblemParser
from PLOI.planning import Planner, PlanningTimeout


class PDDLPlanner(Planner):
    """An abstract PDDL planner for PDDLGym.
    """
    def __call__(self, domain, state, timeout):
        act_preds = [domain.predicates[a] for a in list(domain.actions)]
        act_space = LiteralSpace(
            act_preds, type_to_parent_types=domain.type_to_parent_types)
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
        domain.write(dom_file)
        lits = set(state.literals)
        if not domain.operators_as_actions:
            lits |= set(act_space.all_ground_literals(state, valid_only=False))
        PDDLProblemParser.create_pddl_file(
            prob_file, state.objects, lits, "myproblem",
            domain.domain_name, state.goal, fast_downward_order=True)
        cmd_str = self._get_cmd_str(dom_file, prob_file, timeout)
        start_time = time.time()
        output = subprocess.getoutput(cmd_str)
        self._cleanup()
        os.remove(dom_file)
        os.remove(prob_file)
        if time.time()-start_time > timeout:
            raise PlanningTimeout("Planning timed out!")
        pddl_plan = self._output_to_plan(output)
        plan = [self._plan_step_to_action(domain, state, act_preds, plan_step)
                for plan_step in pddl_plan]
        return plan

    @abc.abstractmethod
    def _get_cmd_str(self, dom_file, prob_file, timeout):
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _output_to_plan(self, output):
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _cleanup(self):
        raise NotImplementedError("Override me!")

    @staticmethod
    def _plan_step_to_action(domain, state, act_predicates, plan_step):
        return parse_plan_step(
            plan_step,
            domain.operators.values(),
            act_predicates,
            state.objects,
            operators_as_actions=domain.operators_as_actions,
        )
