#!/bin/bash

trap "exit" INT
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR
rm -rf results
mkdir results

#TRAINING
tmuxp load train_dataset2d.yaml
tmuxp load train_dataset3d.yaml

#TESTING, one by one to get a more accurate computation time
cd ..

python main.py --method walker_1 --dimension_dataset 2  --weight_soft_cost 0    --train False
python main.py --method UU       --dimension_dataset 2  --weight_soft_cost 0    --train False
python main.py --method UU       --dimension_dataset 2  --weight_soft_cost 10   --train False
python main.py --method UU       --dimension_dataset 2  --weight_soft_cost 100  --train False
python main.py --method UU       --dimension_dataset 2  --weight_soft_cost 1000 --train False
python main.py --method UP       --dimension_dataset 2  --weight_soft_cost 0    --train False
python main.py --method UP       --dimension_dataset 2  --weight_soft_cost 10   --train False
python main.py --method UP       --dimension_dataset 2  --weight_soft_cost 100  --train False
python main.py --method UP       --dimension_dataset 2  --weight_soft_cost 1000 --train False
python main.py --method DC3      --dimension_dataset 2  --weight_soft_cost 0    --train False
python main.py --method DC3      --dimension_dataset 2  --weight_soft_cost 10   --train False
python main.py --method DC3      --dimension_dataset 2  --weight_soft_cost 100  --train False
python main.py --method DC3      --dimension_dataset 2  --weight_soft_cost 1000 --train False
python main.py --method PP       --dimension_dataset 2  --weight_soft_cost 0    --train False
python main.py --method Bar      --dimension_dataset 2  --weight_soft_cost 0    --train False


#Note that the Bar method cannot be used with quadratic constraints
python main.py --method walker_1 --dimension_dataset 3  --weight_soft_cost 0    --train False
python main.py --method UU       --dimension_dataset 3  --weight_soft_cost 0    --train False
python main.py --method UU       --dimension_dataset 3  --weight_soft_cost 10   --train False
python main.py --method UU       --dimension_dataset 3  --weight_soft_cost 100  --train False
python main.py --method UU       --dimension_dataset 3  --weight_soft_cost 1000 --train False
python main.py --method UP       --dimension_dataset 3  --weight_soft_cost 0    --train False
python main.py --method UP       --dimension_dataset 3  --weight_soft_cost 10   --train False
python main.py --method UP       --dimension_dataset 3  --weight_soft_cost 100  --train False
python main.py --method UP       --dimension_dataset 3  --weight_soft_cost 1000 --train False
python main.py --method DC3      --dimension_dataset 3  --weight_soft_cost 0    --train False
python main.py --method DC3      --dimension_dataset 3  --weight_soft_cost 10   --train False
python main.py --method DC3      --dimension_dataset 3  --weight_soft_cost 100  --train False
python main.py --method DC3      --dimension_dataset 3  --weight_soft_cost 1000 --train False
python main.py --method PP       --dimension_dataset 3  --weight_soft_cost 0 --train False


cd $SCRIPT_DIR
python merge_all_results.py