"""Top-level code for PLOI.
"""

import os
import time
import argparse
import pddlgym
from PLOI.planning import PlanningTimeout, PlanningFailure, FD, \
    validate_strips_plan, IncrementalPlanner
from PLOI.guidance import NoSearchGuidance, GNNSearchGuidance


def _test_planner(planner, domain_name, num_problems, timeout):
    print("Running testing...")
    env = pddlgym.make("PDDLEnv{}-v0".format(domain_name))
    num_problems = min(num_problems, len(env.problems))
    for problem_idx in range(num_problems):
        print("\tTesting problem {} of {}".format(problem_idx+1, num_problems),
              flush=True)
        env.fix_problem_index(problem_idx)
        state, _ = env.reset()
        start = time.time()
        try:
            plan = planner(env.domain, state, timeout=timeout)
        except (PlanningTimeout, PlanningFailure) as e:
            print("\t\tPlanning failed with error: {}".format(e), flush=True)
            continue
        # Validate plan on the full test problem.
        if not validate_strips_plan(
                domain_file=env.domain.domain_fname,
                problem_file=env.problems[problem_idx].problem_fname,
                plan=plan):
            print("\t\tPlanning returned an invalid plan")
            continue
        print("\t\tSuccess, got plan of length {} in {:.5f} seconds".format(
            len(plan), time.time()-start), flush=True)


def _create_planner(planner_name):
    if planner_name == "fd-lama-first":
        return FD(alias_flag="--alias lama-first")
    if planner_name == "fd-opt-lmcut":
        return FD(alias_flag="--alias seq-opt-lmcut")
    raise Exception("Unrecognized planner name '{}'.".format(planner_name))


def _create_guider(guider_name, planner_name, num_train_problems,
                   is_strips_domain, num_epochs, seed):
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if guider_name == "no-guidance":
        return NoSearchGuidance()
    if guider_name == "gnn-bce-10":
        planner = _create_planner(planner_name)
        return GNNSearchGuidance(
            training_planner=planner,
            num_train_problems=num_train_problems,
            num_epochs=num_epochs,
            criterion_name="bce",
            bce_pos_weight=10,
            load_from_file=True,
            load_dataset_from_file=True,
            dataset_file_prefix=os.path.join(model_dir, "training_data"),
            save_model_prefix=os.path.join(
                model_dir, "bce10_model_last_seed{}".format(seed)),
            is_strips_domain=is_strips_domain,
        )
    raise Exception("Unrecognized guider name '{}'.".format(guider_name))


def _run(domain_name, train_planner_name, test_planner_name,
         guider_name, num_seeds, num_train_problems, num_test_problems,
         do_incremental_planning, timeout, num_epochs):
    print("Starting run:")
    print("\tDomain: {}".format(domain_name))
    print("\tTrain planner: {}".format(train_planner_name))
    print("\tTest planner: {}".format(test_planner_name))
    print("\tGuider: {}".format(guider_name))
    print("\tDoing incremental planning? {}".format(do_incremental_planning))
    print("\t{} seeds, {} train problems, {} test problems".format(
        num_seeds, num_train_problems, num_test_problems), flush=True)
    print("\n\n")

    if not do_incremental_planning:
        assert guider_name == "no-guidance", "Cannot guide non-incremental!"

    planner = _create_planner(test_planner_name)
    pddlgym_env_names = {"Blocks": "Manyblockssmallpiles",
                         "Miconic": "Manymiconic",
                         "Gripper": "Manygripper",
                         "Ferry": "Manyferry",
                         "Logistics": "Manylogistics",
                         "Hanoi": "Hanoi_operator_actions"}
    assert domain_name in pddlgym_env_names
    domain_name = pddlgym_env_names[domain_name]
    is_strips_domain = True

    for seed in range(num_seeds):
        print("Starting seed {}".format(seed), flush=True)

        guider = _create_guider(guider_name, train_planner_name,
                                num_train_problems, is_strips_domain,
                                num_epochs, seed)
        guider.seed(seed)
        guider.train(domain_name)

        if do_incremental_planning:
            planner_to_test = IncrementalPlanner(
                is_strips_domain=is_strips_domain,
                base_planner=planner, search_guider=guider, seed=seed)
        else:
            planner_to_test = planner

        _test_planner(planner_to_test, domain_name+"Test",
                      num_problems=num_test_problems, timeout=timeout)
    print("\n\nFinished run\n\n\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", required=True, type=str)
    parser.add_argument("--train_planner_name", type=str, default="")
    parser.add_argument("--test_planner_name", required=True, type=str)
    parser.add_argument("--guider_name", required=True, type=str)
    parser.add_argument("--num_seeds", required=True, type=int)
    parser.add_argument("--num_train_problems", type=int, default=0)
    parser.add_argument("--num_test_problems", required=True, type=int)
    parser.add_argument("--do_incremental_planning", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--num_epochs", type=int, default=1001)
    args = parser.parse_args()

    _run(args.domain_name, args.train_planner_name,
         args.test_planner_name, args.guider_name, args.num_seeds,
         args.num_train_problems, args.num_test_problems,
         args.do_incremental_planning, args.timeout, args.num_epochs)
