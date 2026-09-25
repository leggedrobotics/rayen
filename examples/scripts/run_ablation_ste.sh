#!/bin/bash

# Ablation of Section 6.4 of the paper: RAYEN vs. RAYEN_STE (stop-gradient / straight-through estimator variant)
# Both methods are trained and tested on Optimization 2 (corridor_dim3.mat)

trap "exit" INT
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR
mkdir -p results

cd ..

#TRAINING (both methods in parallel)
python main.py --method RAYEN      --dimension_dataset 3  --weight_soft_cost 0     --test False &
python main.py --method RAYEN_STE  --dimension_dataset 3  --weight_soft_cost 0     --test False &
wait

#TESTING, one by one to get a more accurate computation time
#We use the -O flag to remove the asserts, see https://docs.python.org/3/using/cmdline.html#cmdoption-O
python -O main.py --method RAYEN      --dimension_dataset 3  --weight_soft_cost 0     --train False
python -O main.py --method RAYEN_STE  --dimension_dataset 3  --weight_soft_cost 0     --train False

#Print the losses normalized with the globally-optimal one (the ones reported in the table of Section 6.4)
cd $SCRIPT_DIR/results
python -c "
import pandas as pd
for method in ['RAYEN', 'RAYEN_STE']:
    df=pd.read_pickle('dataset3d_'+method+'_weight_soft_cost_0.0.pkl').set_index('method')
    opt=df.loc['dataset3d_Optimization']
    res=df.loc['dataset3d_'+method+'_weight_soft_cost_0.0']
    print(f\"{method:10s} [In dist] n.loss={res['[In dist] loss']/opt['[In dist] loss']:.4f}, violation={res['[In dist] violation']:.5f} | [Out dist] n.loss={res['[Out dist] loss']/opt['[Out dist] loss']:.4f}, violation={res['[Out dist] violation']:.5f}\")
"
