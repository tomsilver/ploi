"""An incremental planner that samples more and more objects until
it finds a plan.
"""

import time
import tempfile
import numpy as np
from pddlgym.structs import State
from pddlgym.spaces import LiteralSpace
from pddlgym.parser import PDDLProblemParser
from PLOI.planning import Planner, PlanningFailure, validate_strips_plan


class IncrementalPlanner(Planner):
    """Sample objects by incrementally lowering a score threshold.
    """
    def __init__(self, is_strips_domain, base_planner, search_guider, seed,
                 gamma=0.9, # parameter for incrementing by score
                 max_iterations=1000,
                 force_include_goal_objects=True):
        super().__init__()
        assert isinstance(base_planner, Planner)
        print("Initializing {} with base planner {}, "
              "guidance {}".format(self.__class__.__name__,
                                   base_planner.__class__.__name__,
                                   search_guider.__class__.__name__))
        self._is_strips_domain = is_strips_domain
        self._gamma = gamma
        self._max_iterations = max_iterations
        self._planner = base_planner
        self._guidance = search_guider
        self._rng = np.random.RandomState(seed=seed)
        self._force_include_goal_objects = force_include_goal_objects

    def __call__(self, domain, state, timeout):
        act_preds = [domain.predicates[a] for a in list(domain.actions)]
        act_space = LiteralSpace(
            act_preds, type_to_parent_types=domain.type_to_parent_types)
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
        domain.write(dom_file)
        lits = set(state.literals)
        if not domain.operators_as_actions:
            lits |= set(act_space.all_ground_literals(
                state, valid_only=False))
        PDDLProblemParser.create_pddl_file(
            prob_file, state.objects, lits, "myproblem",
            domain.domain_name, state.goal, fast_downward_order=True)
        cur_objects = set()
        start_time = time.time()
        if self._force_include_goal_objects:
            # Always start off considering objects in the goal.
            for lit in state.goal.literals:
                cur_objects |= set(lit.variables)
        # Get scores once.
        object_to_score = {obj: self._guidance.score_object(obj, state)
                           for obj in state.objects if obj not in cur_objects}
        # Initialize threshold.
        threshold = self._gamma
        for _ in range(self._max_iterations):
            # Find new objects by incrementally lowering threshold.
            unused_objs = sorted(list(state.objects-cur_objects))
            new_objs = set()
            while unused_objs:
                # Geometrically lower threshold.
                threshold *= self._gamma
                # See if there are any new objects.
                new_objs = {o for o in unused_objs
                            if object_to_score[o] > threshold}
                # If there are, try planning with them.
                if new_objs:
                    break
            cur_objects |= new_objs
            # Keep only literals referencing currently considered objects.
            cur_lits = set()
            for lit in state.literals:
                if all(var in cur_objects for var in lit.variables):
                    cur_lits.add(lit)
            dummy_state = State(cur_lits, cur_objects, state.goal)
            # Try planning with only this object set.
            print("[Trying to plan with {} objects of {} total, "
                  "threshold is {}...]".format(
                      len(cur_objects), len(state.objects), threshold),
                  flush=True)
            try:
                time_elapsed = time.time()-start_time
                # Get a plan from base planner & validate it.
                plan = self._planner(domain, dummy_state, timeout-time_elapsed)
                if not validate_strips_plan(domain_file=dom_file,
                                            problem_file=prob_file,
                                            plan=plan):
                    raise PlanningFailure("Invalid plan")
            except PlanningFailure:
                # Try again with more objects.
                if len(cur_objects) == len(state.objects):
                    # We already tried with all objects, give up.
                    break
                continue
            return plan
        raise PlanningFailure("Plan not found! Reached max_iterations.")
