"""Fast-downward planner.
See information at: http://www.fast-downward.org/ObtainingAndRunningFastDownward
"""

import re
import os
import sys
import subprocess
import tempfile
from PLOI.planning import PDDLPlanner, PlanningFailure

FD_URL = "https://github.com/ronuchit/downward.git"


class FD(PDDLPlanner):
    """Fast-downward planner.
    """
    def __init__(self, alias_flag):
        super().__init__()
        dirname = os.path.dirname(os.path.realpath(__file__))
        self._exec = os.path.join(dirname, "FD/fast-downward.py")
        assert alias_flag in ("--alias lama-first",
                              "--alias seq-opt-lmcut")
        if alias_flag == "--alias seq-opt-lmcut":
            print("Instantiating FD in OPTIMAL mode")
        else:
            print("Instantiating FD in SATISFICING mode")
        self._alias_flag = alias_flag
        if not os.path.exists(self._exec):
            self._install_fd()

    def _get_cmd_str(self, dom_file, prob_file, timeout):
        sas_file = tempfile.NamedTemporaryFile(delete=False).name
        timeout_cmd = "gtimeout" if sys.platform == "darwin" else "timeout"
        cmd_str = "{} {} {} {} --sas-file {} {} {}".format(
            timeout_cmd, timeout, self._exec, self._alias_flag,
            sas_file, dom_file, prob_file)
        return cmd_str

    def _output_to_plan(self, output):
        if "Solution found!" not in output:
            raise PlanningFailure("Plan not found with FD! Error: {}".format(
                output))
        fd_plan = re.findall(r"(.+) \(\d+?\)", output.lower())
        if not fd_plan:
            raise PlanningFailure("Plan not found with FD! Error: {}".format(
                output))
        return fd_plan

    def _cleanup(self):
        cmd_str = "{} --cleanup".format(self._exec)
        subprocess.getoutput(cmd_str)

    def _install_fd(self):
        loc = os.path.dirname(self._exec)
        # Install and compile FD.
        os.system("git clone {} {}".format(FD_URL, loc))
        os.system("cd {} && ./build.py && cd -".format(loc))
        assert os.path.exists(self._exec)
