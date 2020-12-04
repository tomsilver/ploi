#!/bin/bash

DOMAIN_NAME="Blocks"
# DOMAIN_NAME="Miconic"
# DOMAIN_NAME="Gripper"
# DOMAIN_NAME="Ferry"
# DOMAIN_NAME="Logistics"
# DOMAIN_NAME="Hanoi"

METHOD="PLOI"
# METHOD="Pure Plan"
# METHOD="Rand Score"

if [ "$METHOD" = "PLOI" ]
then
    cmd_str="python main.py --domain_name $DOMAIN_NAME --train_planner_name fd-opt-lmcut --test_planner_name fd-lama-first --guider_name gnn-bce-10 --num_seeds 10 --num_train_problems 40 --num_test_problems 10 --do_incremental_planning 1 --timeout 120"
fi

if [ "$METHOD" = "Pure Plan" ]
then
    cmd_str="python main.py --domain_name $DOMAIN_NAME --test_planner_name fd-lama-first --guider_name no-guidance --num_seeds 10 --num_test_problems 10 --do_incremental_planning 0 --timeout 120"
fi

if [ "$METHOD" = "Rand Score" ]
then
    cmd_str="python main.py --domain_name $DOMAIN_NAME --test_planner_name fd-lama-first --guider_name no-guidance --num_seeds 10 --num_test_problems 10 --do_incremental_planning 1 --timeout 120"
fi

echo "Running Python command:"
echo $cmd_str
eval $cmd_str
