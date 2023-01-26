"""Validate plans.
"""

import tempfile
import os
import subprocess

VALIDATE_CMD = "validate"


def verify_validate_installed():
    """Return True if validate is installed properly."""
    cmd_str = VALIDATE_CMD
    output = subprocess.getoutput(cmd_str)
    return "VAL: The PDDL+ plan validation tool" in output


def validate_strips_plan(domain_file, problem_file, plan):
    """Return True for a successfully validated plan, False otherwise.
    """
    plan_str = ""
    for t, action in enumerate(plan):
        plan_str += "{}: {}\n".format(t, action.pddl_str())
    planfile = tempfile.NamedTemporaryFile(delete=False).name
    with open(planfile, "w") as f:
        f.write(plan_str)
    cmd_str = "{} -v {} {} {}".format(VALIDATE_CMD, domain_file, problem_file, planfile)
    output = subprocess.getoutput(cmd_str)
    os.remove(planfile)
    if "Plan valid" in output:
        return True
    return False
