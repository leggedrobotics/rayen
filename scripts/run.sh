#!/bin/bash

trap "exit" INT
set -e

# rm -rf results
# mkdir results

#TRAINING
# tmuxp load ./mysession.yaml


#TESTING, one by one to get a better estimation of the computation time
cd ..
python main.py --method walker_1   --weight_soft_cost 0    --train False
python main.py --method Bar        --weight_soft_cost 0    --train False
python main.py --method UU         --weight_soft_cost 0    --train False
python main.py --method UU         --weight_soft_cost 10   --train False
python main.py --method UU         --weight_soft_cost 100  --train False
python main.py --method UU         --weight_soft_cost 1000 --train False
python main.py --method UP         --weight_soft_cost 0    --train False
python main.py --method UP         --weight_soft_cost 10   --train False
python main.py --method UP         --weight_soft_cost 100  --train False
python main.py --method UP         --weight_soft_cost 1000 --train False
python main.py --method DC3        --weight_soft_cost 0    --train False
python main.py --method DC3        --weight_soft_cost 10   --train False
python main.py --method DC3        --weight_soft_cost 100  --train False
python main.py --method DC3        --weight_soft_cost 1000 --train False



python merge_all_results.py