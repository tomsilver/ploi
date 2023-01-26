# PLOI

This repository houses code for the AAAI 2021 paper:

Planning with Learned Object Importance in Large Problem Instances using Graph Neural Networks

Tom Silver*, Rohan Chitnis*, Aidan Curtis,  Joshua Tenenbaum, Tomas Lozano-Perez, Leslie Pack Kaelbling.

For any questions or issues with the code, please email ronuchit@mit.edu and tslvr@mit.edu.

Link to paper: https://arxiv.org/abs/2009.05613

Link to video: https://www.youtube.com/watch?v=FWsVJc2fvCE

Instructions for running (tested on Mac and Linux):
* Use Python 3.6 or higher.
* Download Python dependencies: `pip install -r requirements.txt`.
* Make sure the directory containing this code is in your PYTHONPATH.
* Download and build the plan validation tool available at https://github.com/KCL-Planning/VAL, then make a symlink called `validate` on your path that points to the `build/Validate` binary, e.g. `ln -s <path to VAL>/build/Validate /usr/local/bin/validate`. If done successfully, running `validate` on your command line should give an output that starts with the line: "VAL: The PDDL+ plan validation tool". If you have trouble with the symlink, you can just directly change `VALIDATE_CMD` in `planning/validate.py` to point to the `build/Validate` binary.

Now, `./run.sh` should work. Different domains and methods can be run by modifying the variables at the top of run.sh.
