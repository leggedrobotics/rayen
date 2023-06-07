#!/bin/bash

pkill -f tmux


trap "exit" INT
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR
rm -rf results
mkdir results

#TRAINING
tmuxp load train.yaml

#TESTING, one by one to get a more accurate computation time
cd ..

#We use the -O flag to remove the asserts, see https://docs.python.org/3/using/cmdline.html#cmdoption-O
python -O main.py --method RAYEN     --dimension_dataset 2  --weight_soft_cost 0     --train False
python -O main.py --method RAYEN_old --dimension_dataset 2  --weight_soft_cost 0     --train False
python -O main.py --method UU        --dimension_dataset 2  --weight_soft_cost 0     --train False
python -O main.py --method UU        --dimension_dataset 2  --weight_soft_cost 10    --train False
python -O main.py --method UU        --dimension_dataset 2  --weight_soft_cost 100   --train False
python -O main.py --method UU        --dimension_dataset 2  --weight_soft_cost 1000  --train False
python -O main.py --method UU        --dimension_dataset 2  --weight_soft_cost 10000 --train False
python -O main.py --method UP        --dimension_dataset 2  --weight_soft_cost 0     --train False
python -O main.py --method UP        --dimension_dataset 2  --weight_soft_cost 10    --train False
python -O main.py --method UP        --dimension_dataset 2  --weight_soft_cost 100   --train False
python -O main.py --method UP        --dimension_dataset 2  --weight_soft_cost 1000  --train False
python -O main.py --method UP        --dimension_dataset 2  --weight_soft_cost 10000 --train False
python -O main.py --method DC3       --dimension_dataset 2  --weight_soft_cost 0     --train False
python -O main.py --method DC3       --dimension_dataset 2  --weight_soft_cost 10    --train False
python -O main.py --method DC3       --dimension_dataset 2  --weight_soft_cost 100   --train False
python -O main.py --method DC3       --dimension_dataset 2  --weight_soft_cost 1000  --train False
python -O main.py --method DC3       --dimension_dataset 2  --weight_soft_cost 10000 --train False
python -O main.py --method PP        --dimension_dataset 2  --weight_soft_cost 0     --train False
python -O main.py --method Bar       --dimension_dataset 2  --weight_soft_cost 0     --train False


#Note that the Bar method cannot be used with quadratic constraints
python -O main.py --method RAYEN     --dimension_dataset 3  --weight_soft_cost 0     --train False
python -O main.py --method RAYEN_old --dimension_dataset 3  --weight_soft_cost 0     --train False
python -O main.py --method UU        --dimension_dataset 3  --weight_soft_cost 0     --train False
python -O main.py --method UU        --dimension_dataset 3  --weight_soft_cost 10    --train False
python -O main.py --method UU        --dimension_dataset 3  --weight_soft_cost 100   --train False
python -O main.py --method UU        --dimension_dataset 3  --weight_soft_cost 1000  --train False
python -O main.py --method UU        --dimension_dataset 3  --weight_soft_cost 10000 --train False
python -O main.py --method UP        --dimension_dataset 3  --weight_soft_cost 0     --train False
python -O main.py --method UP        --dimension_dataset 3  --weight_soft_cost 10    --train False
python -O main.py --method UP        --dimension_dataset 3  --weight_soft_cost 100   --train False
python -O main.py --method UP        --dimension_dataset 3  --weight_soft_cost 1000  --train False
python -O main.py --method UP        --dimension_dataset 3  --weight_soft_cost 10000 --train False
python -O main.py --method DC3       --dimension_dataset 3  --weight_soft_cost 0     --train False
python -O main.py --method DC3       --dimension_dataset 3  --weight_soft_cost 10    --train False
python -O main.py --method DC3       --dimension_dataset 3  --weight_soft_cost 100   --train False
python -O main.py --method DC3       --dimension_dataset 3  --weight_soft_cost 1000  --train False
python -O main.py --method DC3       --dimension_dataset 3  --weight_soft_cost 10000 --train False
python -O main.py --method PP        --dimension_dataset 3  --weight_soft_cost 0     --train False


cd $SCRIPT_DIR
python merge_all_results.py

python -O time_analysis.py