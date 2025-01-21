# Smart Initial Basis Selection for Linear Programs

### This notebook contains the implmenetation of the paper 'Smart Initial Basis Selection for Linear Programs' published in ICML2023.
https://openreview.net/forum?id=Mha86sOok1

## Overview 

- Background: Companies or large organizations often need to repeatedly solve similar linear programming (LP) problems, such as scheduling a large number of flights per hour at an airport.
- Problem Definition: Given a set of historical LPs and a similar test LP, the goal is to predict the initial basis for the test LP that allows the simplex algorithm to converge faster starting from that basis.

    - Consider the following standard form of LP: $$\begin{align*}
\text{Minimize } & c^T x \\
\text{Subject to:} \\
& Ax = s \\
& l^x \leq x \leq u^x \\
& l^s \leq s \leq u^s
\end{align*}$$

    - Note: Lower bounds $l^x$, $l^s$ can be $-\infty$, and upper bounds $u^x$, $u^s$ can be $\infty$.

- Algorithm Steps:

    - Represent the LP as a bipartite graph and design features for each variable and constraint.
    - Build a training set using historical LPs and train a two-layer graph neural network.
    - Use knowledge masking to resolve conflicts between the predicted basis by the graph network and the variable bounds.
    - Generate and adjust the initial basis to ensure compliance with the rules of a valid basis.

## Prerequisites

- This demo is tested with Ubuntu 20.04, Python(>=3.8 & <=3.10), PyTorch(>=1.8 & <=1.10) and HiGHS 1.3.1, and are generalizable to other linux system. 


```python
!wget https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/lp-gnn/code.zip
!unzip -qo code.zip 
```


```python
import os 
## setup for network if needed 
# os.environ['http_proxy']=os.environ['https_proxy']="http://127.0.0.1:3128"
# os.environ['home']='/home/xinglu/' 

## The most recommened way is through conda
## because compiling pytorch-sparse from source code fails if libarary is missing or gcc version mismatch
## Note: change to pytorch with GPU if needed
!conda install pytorch-sparse==0.6.12 pytorch-scatter pytorch cpuonly -c pytorch -c pyg -y
!pip install "torch_geometric==2.1"
## If conda failed, try install with pip with following:
# !pip install torch==1.8.0
# import re
# pv = !python --version
# version_string=pv[0] 
# major_version = re.search(r'(\d+)\.', version_string).group(1)
# minor_version = re.search(r'\.(\d+)\.', version_string).group(1)
# vsm=vs=f"{major_version}{minor_version}"
# if vs=='37': vsm=vs+'m'
# !rm -rf *.whl *.whl.*
# url=f"https://data.pyg.org/whl/torch-1.8.0%2Bcpu/torch_sparse-0.6.12-cp{vs}-cp{vsm}-linux_x86_64.whl"
# !wget $url --no-check-certificate
# url=f"https://data.pyg.org/whl/torch-1.8.0%2Bcpu/torch_scatter-2.0.8-cp{vs}-cp{vsm}-linux_x86_64.whl"
# !wget $url --no-check-certificate
# !pip install ./*.whl
```


```python
%%capture 
!pip install pandas numpy
!pip install mip tables 
!pip install colorlog msgpack msgpack_numpy tensorboard easydict seaborn 
```


```python
%%capture 
## Before starting to train and test, the experiment folder `highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3` for `mirp` is uploaded under runs, we can extract and show the performance
exp_nm="highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3"
!python scripts/extract_time.py --exp_nm $exp_nm

import torch_geometric
import numpy as np 
from utils import df_load, check_df, filter_cols, proc

df=df_load(f"runs/{exp_nm}/time.h5")
df=df[df.split=='val']
if df.isnull().any().any(): print('warn: ori contain nan')
df.index=df.fn
check_vals=[np.inf,-1,-2,-3]
for check_val in check_vals:
    if check_df(df,check_val).shape[0]!=0: print('warn: table contains errorcode', check_val)  
df=df.replace(check_vals,np.nan)  
dft=df.describe().loc[['mean','std']]
use_method=f'gnn-bas-0'
cs=filter_cols([
            f'highs-no-bas/niter', f'highs-ca-bas/niter', f'{use_method}/niter',            
            ], df.columns)
res=dft[cs].apply(proc, axis=0).to_frame().T
```


```python
res
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>highs-no-bas/niter</th>
      <th>highs-ca-bas/niter</th>
      <th>gnn-bas-0/niter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>$29.9K{\scriptscriptstyle  \pm 17.0K}$</td>
      <td>$26.2K{\scriptscriptstyle  \pm 14.2K}$</td>
      <td>$17.4K{\scriptscriptstyle  \pm 11.0K}$</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%capture
## Compile and Install HiGHS (https://ergo-code.github.io/HiGHS/get-started). Add the path contain exective `highs` to system enviroment variable PATH. 
## We cutomized the HiGHS code such that the log file contains more information. This part code comes with this notebook. 
## Ubuntu 20.04 pairs with g++ 9.4 and cmake 3.16. Make sure cmake and g++ are installed, e.g., via apt on Ubuntu. 
!cd HiGHS-master&&mkdir build -p&&cd build&&cmake .. && make -j10 
cur=%pwd
PATH=os.environ['PATH']
os.environ['PATH']=f"{cur}/HiGHS-master/build/bin:{PATH}" 
```


```python
## double check that highs is successfully installed
!highs -h
```

    xinglu: 1
    HiGHS options
    Usage:
      highs [OPTION...] [file]
    
          --model_file arg        File of model to solve.
          --presolve arg          Presolve: "choose" by default - "on"/"off" are
                                  alternatives.
          --solver arg            Solver: "choose" by default - "simplex"/"ipm"
                                  are alternatives.
          --parallel arg          Parallel solve: "choose" by default -
                                  "on"/"off" are alternatives.
          --time_limit arg        Run time limit (seconds - double).
          --options_file arg      File containing HiGHS options.
          --solution_file arg     File for writing out model solution.
          --write_model_file arg  File for writing out model.
          --random_seed arg       Seed to initialize random number generation.
          --ranging arg           Compute cost, bound, RHS and basic solution
                                  ranging.
      -h, --help                  Print help.
    


## Step 1: data preparation

- `python run_prep_data.py` will prepare the dataset, including running Simplex algorithm, saving the log, and serialize the dataset for training. 
    - In this demo we use mirp dataset from https://mirplib.scl.gatech.edu/ (Group 1, https://mirplib.scl.gatech.edu/sites/default/files/Group1_MPS_files.zip), which already comes with this demo. 


```python
!python run_prep_data.py --dataset mirp
```

     2023-05-22 16:45:35,719 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 16:45:35,719 utils.py [line:103] [32mINFO[0m cmd is pwd[0m
     2023-05-22 16:45:35,728 utils.py [line:117] [32mINFO[0m stdout | /home/xinglu/Downloads/lp-gnn.release
    [0m
    proced args timer : <utils.Timer object at 0x7fdefc6e0130> 
    root_path : /home/xinglu/Downloads/lp-gnn.release 
    home_path : /home/xinglu 
    prefetch_factor : 4 
    ori_args : {'dev': 0, 'exp_nm': 'tmp', 'opt': 'adam', 'lr': 0.001, 'epochs': 30, 'arch': 'GCN_FC(8,8,hids=128)', 'seed': 0, 'num_workers': 8, 'batch_size': 327680, 'load_from': 'None', 'dataset': 'mirp', 'solver_prefix': 'highs-', 'data_prefix': '/home/xinglu/Downloads/lp-gnn.release/lp-dataset/', 'exist_ok': 1, 'edge_num_thresh': 12000000.0, 'verbose': 1, 'skip_exist': 1, 'f': None, 'init_method': None, 'chunk': '0/1', 'loss': 'balanced', 'inference_manager': 'InferenceManager(0,)', 'lp_method': '1', 'gW': None, 'split': 'val', 'log_prefix': '/home/xinglu/Downloads/lp-gnn.release/runs/', 'fp16': 0} 
    dev : cpu 
    exp_nm : tmp 
    opt : adam 
    lr : 0.001 
    epochs : 30 
    arch : GCN_FC(8,8,hids=128) 
    seed : 0 
    num_workers : 8 
    batch_size : 327680 
    load_from : None 
    dataset : mirp 
    solver_prefix : highs- 
    data_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset/ 
    exist_ok : 1 
    edge_num_thresh : 12000000.0 
    verbose : 1 
    skip_exist : 1 
    f : None 
    init_method : None 
    chunk : 0/1 
    loss : balanced 
    inference_manager : InferenceManager(0,) 
    lp_method : 1 
    gW : None 
    split : val 
    log_prefix : /home/xinglu/Downloads/lp-gnn.release/runs/ 
    fp16 : 0 
    dataset_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp 
    dataset_processed_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-inp_tgt/ 
    log_dir : /home/xinglu/Downloads/lp-gnn.release/runs/tmp/ 
    
     2023-05-22 16:45:35,732 utils.py [line:486] [32mINFO[0m stream logging set up [0m
    ------------------------------
     2023-05-22 16:45:35,732 utils.py [line:83] [32mINFO[0m cmd:python scripts/run_solver.py --dev 0 --exp_nm tmp --opt adam --lr 0.001 --epochs 30 --arch GCN_FC(8,8,hids=128) --seed 0 --num_workers 8 --batch_size 327680 --load_from None --dataset mirp --solver_prefix highs- --data_prefix /home/xinglu/Downloads/lp-gnn.release/lp-dataset/ --exist_ok 1 --edge_num_thresh 12000000.0 --verbose 1 --skip_exist 1 --chunk 0/1 --loss balanced --inference_manager InferenceManager(0,) --lp_method 1 --gW None --split val --log_prefix /home/xinglu/Downloads/lp-gnn.release/runs/ --fp16 0  --num_workers 1 [0m
     2023-05-22 16:45:37,035 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 16:45:37,035 utils.py [line:103] [32mINFO[0m cmd is pwd[0m
     2023-05-22 16:45:37,045 utils.py [line:117] [32mINFO[0m stdout | /home/xinglu/Downloads/lp-gnn.release
    [0m
    proced args timer : <utils.Timer object at 0x7f3f041986d0> 
    root_path : /home/xinglu/Downloads/lp-gnn.release 
    home_path : /home/xinglu 
    prefetch_factor : 4 
    ori_args : {'dev': 0, 'exp_nm': 'tmp', 'opt': 'adam', 'lr': 0.001, 'epochs': 30, 'arch': 'GCN_FC(8,8,hids=128)', 'seed': 0, 'num_workers': 1, 'batch_size': 327680, 'load_from': 'None', 'dataset': 'mirp', 'solver_prefix': 'highs-', 'data_prefix': '/home/xinglu/Downloads/lp-gnn.release/lp-dataset/', 'exist_ok': 1, 'edge_num_thresh': 12000000.0, 'verbose': 1, 'skip_exist': 1, 'f': None, 'init_method': None, 'chunk': '0/1', 'loss': 'balanced', 'inference_manager': 'InferenceManager(0,)', 'lp_method': '1', 'gW': None, 'split': 'val', 'log_prefix': '/home/xinglu/Downloads/lp-gnn.release/runs/', 'fp16': 0} 
    dev : cpu 
    exp_nm : tmp 
    opt : adam 
    lr : 0.001 
    epochs : 30 
    arch : GCN_FC(8,8,hids=128) 
    seed : 0 
    num_workers : 1 
    batch_size : 327680 
    load_from : None 
    dataset : mirp 
    solver_prefix : highs- 
    data_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset/ 
    exist_ok : 1 
    edge_num_thresh : 12000000.0 
    verbose : 1 
    skip_exist : 1 
    f : None 
    init_method : None 
    chunk : 0/1 
    loss : balanced 
    inference_manager : InferenceManager(0,) 
    lp_method : 1 
    gW : None 
    split : val 
    log_prefix : /home/xinglu/Downloads/lp-gnn.release/runs/ 
    fp16 : 0 
    dataset_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp 
    dataset_processed_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-inp_tgt/ 
    log_dir : /home/xinglu/Downloads/lp-gnn.release/runs/tmp/ 
    
     2023-05-22 16:45:37,048 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 16:45:37,049 utils.py [line:667] [32mINFO[0m chunk: 28 0 28[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_1_DR1_3_VC1_V7a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_1_DR1_3_VC1_V7a_t45.bas
     2023-05-22 16:45:37,108 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_3_VC1_V7a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_3_VC1_V7a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_3_VC1_V7a_t45.bas -ss 1 [0m
     2023-05-22 16:45:38,364 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_3_VC1_V7a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_3_VC1_V7a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_3_VC1_V7a_t45 has 7909 rows; 8040 cols; 27719 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
           8067    -2.5646623198e+04 Pr: 0(0); Du: 0(2.21462e-13) 1s
           8067    -2.5646623198e+04 Pr: 0(0); Du: 0(2.21462e-13) 1s
    Model   status      : Optimal
    Simplex   iterations: 8067
    Objective value     : -2.5646623198e+04
    HiGHS run time      :          1.19
    [0m
     2023-05-22 16:45:38,365 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_1_DR1_3_VC1_V7a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_1_DR1_3_VC1_V7a_t60.bas
     2023-05-22 16:45:38,376 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_3_VC1_V7a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_3_VC1_V7a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_3_VC1_V7a_t60.bas -ss 1 [0m
     2023-05-22 16:45:40,663 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_3_VC1_V7a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_3_VC1_V7a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_3_VC1_V7a_t60 has 10864 rows; 11265 cols; 39089 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          11407    -3.3347591194e+04 Pr: 0(0); Du: 0(2.54124e-12) 2s
          11407    -3.3347591194e+04 Pr: 0(0); Du: 0(2.54124e-12) 2s
    Model   status      : Optimal
    Simplex   iterations: 11407
    Objective value     : -3.3347591194e+04
    HiGHS run time      :          2.20
    [0m
     2023-05-22 16:45:40,664 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_1_DR1_4_VC3_V8a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_1_DR1_4_VC3_V8a_t45.bas
     2023-05-22 16:45:40,674 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V8a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V8a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V8a_t45.bas -ss 1 [0m
     2023-05-22 16:45:42,992 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V8a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V8a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V8a_t45 has 11104 rows; 12396 cols; 42568 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          11216    -2.4336511831e+04 Pr: 0(0); Du: 0(2.17326e-14) 2s
          11216    -2.4336511831e+04 Pr: 0(0); Du: 0(2.17326e-14) 2s
    Model   status      : Optimal
    Simplex   iterations: 11216
    Objective value     : -2.4336511831e+04
    HiGHS run time      :          2.23
    [0m
     2023-05-22 16:45:42,992 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_1_DR1_4_VC3_V8a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_1_DR1_4_VC3_V8a_t60.bas
     2023-05-22 16:45:43,003 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V8a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V8a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V8a_t60.bas -ss 1 [0m
     2023-05-22 16:45:48,332 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V8a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V8a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V8a_t60 has 15334 rows; 17541 cols; 60628 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          17306    -3.1852671848e+04 Pr: 1226(23806.1); Du: 0(2.19465e-08) 5s
          17698    -3.1850671995e+04 Pr: 0(0); Du: 0(1.76058e-12) 5s
          17698    -3.1850671995e+04 Pr: 0(0); Du: 0(1.76058e-12) 5s
    Model   status      : Optimal
    Simplex   iterations: 17698
    Objective value     : -3.1850671995e+04
    HiGHS run time      :          5.21
    [0m
     2023-05-22 16:45:48,332 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_1_DR1_4_VC3_V9a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_1_DR1_4_VC3_V9a_t45.bas
     2023-05-22 16:45:48,343 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V9a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V9a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V9a_t45.bas -ss 1 [0m
     2023-05-22 16:45:51,158 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V9a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V9a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V9a_t45 has 12250 rows; 13394 cols; 46161 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          12437    -2.4784918665e+04 Pr: 0(0); Du: 0(2.28397e-13) 2s
          12437    -2.4784918665e+04 Pr: 0(0); Du: 0(2.28397e-13) 2s
    Model   status      : Optimal
    Simplex   iterations: 12437
    Objective value     : -2.4784918665e+04
    HiGHS run time      :          2.70
    [0m
     2023-05-22 16:45:51,159 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_1_DR1_4_VC3_V9a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_1_DR1_4_VC3_V9a_t60.bas
     2023-05-22 16:45:51,171 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V9a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V9a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V9a_t60.bas -ss 1 [0m
     2023-05-22 16:45:57,722 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V9a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V9a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V9a_t60 has 16990 rows; 19154 cols; 66441 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          16523    -3.2359291863e+04 Pr: 1583(68585); Du: 0(1.91665e-13) 5s
          19518    -3.2300787062e+04 Pr: 0(0); Du: 0(3.45701e-12) 6s
          19518    -3.2300787062e+04 Pr: 0(0); Du: 0(3.45701e-12) 6s
    Model   status      : Optimal
    Simplex   iterations: 19518
    Objective value     : -3.2300787062e+04
    HiGHS run time      :          6.42
    [0m
     2023-05-22 16:45:57,722 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_2_DR1_3_VC2_V6a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_2_DR1_3_VC2_V6a_t45.bas
     2023-05-22 16:45:57,733 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_2_DR1_3_VC2_V6a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_2_DR1_3_VC2_V6a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_2_DR1_3_VC2_V6a_t45.bas -ss 1 [0m
     2023-05-22 16:45:59,010 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_2_DR1_3_VC2_V6a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_2_DR1_3_VC2_V6a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_2_DR1_3_VC2_V6a_t45 has 9306 rows; 9638 cols; 34302 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
           7582    -2.3254484419e+04 Pr: 0(0); Du: 0(4.70332e-13) 1s
           7582    -2.3254484419e+04 Pr: 0(0); Du: 0(4.70332e-13) 1s
    Model   status      : Optimal
    Simplex   iterations: 7582
    Objective value     : -2.3254484419e+04
    HiGHS run time      :          1.21
    [0m
     2023-05-22 16:45:59,011 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_2_DR1_3_VC2_V6a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_2_DR1_3_VC2_V6a_t60.bas
     2023-05-22 16:45:59,021 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_2_DR1_3_VC2_V6a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_2_DR1_3_VC2_V6a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_2_DR1_3_VC2_V6a_t60.bas -ss 1 [0m
     2023-05-22 16:46:02,193 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_2_DR1_3_VC2_V6a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_2_DR1_3_VC2_V6a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_2_DR1_3_VC2_V6a_t60 has 12876 rows; 13553 cols; 48642 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          13031    -3.0588817139e+04 Pr: 0(0); Du: 0(5.55112e-17) 3s
          13031    -3.0588817139e+04 Pr: 0(0); Du: 0(5.55112e-17) 3s
    Model   status      : Optimal
    Simplex   iterations: 13031
    Objective value     : -3.0588817139e+04
    HiGHS run time      :          3.06
    [0m
     2023-05-22 16:46:02,193 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_2_DR1_3_VC3_V8a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_2_DR1_3_VC3_V8a_t45.bas
     2023-05-22 16:46:02,203 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_2_DR1_3_VC3_V8a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_2_DR1_3_VC3_V8a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_2_DR1_3_VC3_V8a_t45.bas -ss 1 [0m
     2023-05-22 16:46:05,570 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_2_DR1_3_VC3_V8a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_2_DR1_3_VC3_V8a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_2_DR1_3_VC3_V8a_t45 has 12325 rows; 12740 cols; 45892 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          13224    -2.8184020119e+04 Pr: 0(0); Du: 0(1.0858e-13) 3s
          13224    -2.8184020119e+04 Pr: 0(0); Du: 0(1.0858e-13) 3s
    Model   status      : Optimal
    Simplex   iterations: 13224
    Objective value     : -2.8184020119e+04
    HiGHS run time      :          3.27
    [0m
     2023-05-22 16:46:05,570 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_2_DR1_3_VC3_V8a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_2_DR1_3_VC3_V8a_t60.bas
     2023-05-22 16:46:05,580 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_2_DR1_3_VC3_V8a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_2_DR1_3_VC3_V8a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_2_DR1_3_VC3_V8a_t60.bas -ss 1 [0m
     2023-05-22 16:46:12,183 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_2_DR1_3_VC3_V8a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_2_DR1_3_VC3_V8a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_2_DR1_3_VC3_V8a_t60 has 17035 rows; 17885 cols; 64912 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          16684    -3.6816250254e+04 Pr: 2714(372666) 5s
          19701    -3.6754344157e+04 Pr: 0(0); Du: 0(2.45817e-13) 6s
          19701    -3.6754344157e+04 Pr: 0(0); Du: 0(2.45817e-13) 6s
    Model   status      : Optimal
    Simplex   iterations: 19701
    Objective value     : -3.6754344157e+04
    HiGHS run time      :          6.47
    [0m
     2023-05-22 16:46:12,184 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_1_DR1_4_VC3_V11a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_1_DR1_4_VC3_V11a_t45.bas
     2023-05-22 16:46:12,194 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V11a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V11a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V11a_t45.bas -ss 1 [0m
     2023-05-22 16:46:15,956 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V11a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V11a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V11a_t45 has 14668 rows; 15762 cols; 54583 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          14808    -2.9768722924e+04 Pr: 0(0); Du: 0(4.24629e-13) 3s
          14808    -2.9768722924e+04 Pr: 0(0); Du: 0(4.24629e-13) 3s
    Model   status      : Optimal
    Simplex   iterations: 14808
    Objective value     : -2.9768722924e+04
    HiGHS run time      :          3.66
    [0m
     2023-05-22 16:46:15,956 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_1_DR1_4_VC3_V11a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_1_DR1_4_VC3_V11a_t60.bas
     2023-05-22 16:46:15,967 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V11a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V11a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V11a_t60.bas -ss 1 [0m
     2023-05-22 16:46:24,564 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V11a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V11a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V11a_t60 has 20428 rows; 22752 cols; 79303 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          16931    -3.8988338992e+04 Pr: 2081(109580); Du: 0(1.02718e-08) 5s
          23023    -3.8849209401e+04 Pr: 0(0); Du: 0(1.09213e-12) 8s
          23023    -3.8849209401e+04 Pr: 0(0); Du: 0(1.09213e-12) 8s
    Model   status      : Optimal
    Simplex   iterations: 23023
    Objective value     : -3.8849209401e+04
    HiGHS run time      :          8.44
    [0m
     2023-05-22 16:46:24,564 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_1_DR1_4_VC3_V12a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_1_DR1_4_VC3_V12a_t45.bas
     2023-05-22 16:46:24,575 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V12a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12a_t45.bas -ss 1 [0m
     2023-05-22 16:46:29,244 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V12a_t45 has 16037 rows; 17440 cols; 60442 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          16689    -3.3450001287e+04 Pr: 0(0); Du: 0(5.20365e-13) 4s
          16689    -3.3450001287e+04 Pr: 0(0); Du: 0(5.20365e-13) 4s
    Model   status      : Optimal
    Simplex   iterations: 16689
    Objective value     : -3.3450001287e+04
    HiGHS run time      :          4.54
    [0m
     2023-05-22 16:46:29,245 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_1_DR1_4_VC3_V12a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_1_DR1_4_VC3_V12a_t60.bas
     2023-05-22 16:46:29,256 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V12a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12a_t60.bas -ss 1 [0m
     2023-05-22 16:46:43,265 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V12a_t60 has 22307 rows; 25045 cols; 87382 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          16704    -4.3845911253e+04 Pr: 3905(7.01991e+06); Du: 0(5.26122e-08) 5s
          24379    -4.3686840654e+04 Pr: 3746(1.96551e+06) 10s
          29840    -4.3576234656e+04 Pr: 0(0); Du: 0(1.60855e-12) 13s
          29840    -4.3576234656e+04 Pr: 0(0); Du: 0(1.60855e-12) 13s
    Model   status      : Optimal
    Simplex   iterations: 29840
    Objective value     : -4.3576234656e+04
    HiGHS run time      :         13.83
    [0m
     2023-05-22 16:46:43,266 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_1_DR1_4_VC3_V12b_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_1_DR1_4_VC3_V12b_t45.bas
     2023-05-22 16:46:43,277 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V12b_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12b_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12b_t45.bas -ss 1 [0m
     2023-05-22 16:46:49,051 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12b_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12b_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V12b_t45 has 16173 rows; 17704 cols; 61418 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          17683    -3.0787875587e+04 Pr: 1782(101256) 5s
          19057    -3.0700543948e+04 Pr: 0(0); Du: 0(1.3517e-13) 5s
          19057    -3.0700543948e+04 Pr: 0(0); Du: 0(1.3517e-13) 5s
    Model   status      : Optimal
    Simplex   iterations: 19057
    Objective value     : -3.0700543948e+04
    HiGHS run time      :          5.65
    [0m
     2023-05-22 16:46:49,052 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR1_1_DR1_4_VC3_V12b_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR1_1_DR1_4_VC3_V12b_t60.bas
     2023-05-22 16:46:49,063 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V12b_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12b_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12b_t60.bas -ss 1 [0m
     2023-05-22 16:47:04,363 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12b_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR1_1_DR1_4_VC3_V12b_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V12b_t60 has 22443 rows; 25309 cols; 88358 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          17162    -4.0351986136e+04 Pr: 5371(2.62064e+06); Du: 0(9.1843e-08) 5s
          24385    -4.0205082198e+04 Pr: 4256(1.54671e+06) 10s
          31323    -4.0065391472e+04 Pr: 0(0); Du: 0(1.68615e-13) 15s
          31323    -4.0065391472e+04 Pr: 0(0); Du: 0(1.68615e-13) 15s
    Model   status      : Optimal
    Simplex   iterations: 31323
    Objective value     : -4.0065391472e+04
    HiGHS run time      :         15.13
    [0m
     2023-05-22 16:47:04,363 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR2_11_DR2_22_VC3_V6a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR2_11_DR2_22_VC3_V6a_t45.bas
     2023-05-22 16:47:04,374 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_11_DR2_22_VC3_V6a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_11_DR2_22_VC3_V6a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_11_DR2_22_VC3_V6a_t45.bas -ss 1 [0m
     2023-05-22 16:47:06,260 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_11_DR2_22_VC3_V6a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_11_DR2_22_VC3_V6a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_11_DR2_22_VC3_V6a_t45 has 10861 rows; 9897 cols; 35921 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          10140    -2.3778931836e+04 Pr: 0(0); Du: 0(1.21321e-12) 1s
          10140    -2.3778931836e+04 Pr: 0(0); Du: 0(1.21321e-12) 1s
    Model   status      : Optimal
    Simplex   iterations: 10140
    Objective value     : -2.3778931836e+04
    HiGHS run time      :          1.81
    [0m
     2023-05-22 16:47:06,261 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR2_11_DR2_22_VC3_V6a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR2_11_DR2_22_VC3_V6a_t60.bas
     2023-05-22 16:47:06,271 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_11_DR2_22_VC3_V6a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_11_DR2_22_VC3_V6a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_11_DR2_22_VC3_V6a_t60.bas -ss 1 [0m
     2023-05-22 16:47:10,632 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_11_DR2_22_VC3_V6a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_11_DR2_22_VC3_V6a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_11_DR2_22_VC3_V6a_t60 has 15271 rows; 14217 cols; 52301 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          15783    -3.1022076544e+04 Pr: 0(0); Du: 0(7.97799e-11) 4s
          15783    -3.1022076544e+04 Pr: 0(0); Du: 0(7.97799e-11) 4s
    Model   status      : Optimal
    Simplex   iterations: 15783
    Objective value     : -3.1022076544e+04
    HiGHS run time      :          4.26
    [0m
     2023-05-22 16:47:10,633 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR2_11_DR2_33_VC4_V11a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR2_11_DR2_33_VC4_V11a_t45.bas
     2023-05-22 16:47:10,644 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_11_DR2_33_VC4_V11a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_11_DR2_33_VC4_V11a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_11_DR2_33_VC4_V11a_t45.bas -ss 1 [0m
     2023-05-22 16:47:22,669 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_11_DR2_33_VC4_V11a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_11_DR2_33_VC4_V11a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_11_DR2_33_VC4_V11a_t45 has 25980 rows; 25411 cols; 93854 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          17181    -3.6438744343e+04 Pr: 4339(1.16905e+06) 5s
          24453    -3.6258127896e+04 Pr: 4237(167943) 10s
          27013    -3.6185656446e+04 Pr: 0(0); Du: 0(1.5464e-13) 11s
          27013    -3.6185656446e+04 Pr: 0(0); Du: 0(1.5464e-13) 11s
    Model   status      : Optimal
    Simplex   iterations: 27013
    Objective value     : -3.6185656446e+04
    HiGHS run time      :         11.85
    [0m
     2023-05-22 16:47:22,670 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR2_11_DR2_33_VC4_V11a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR2_11_DR2_33_VC4_V11a_t60.bas
     2023-05-22 16:47:22,680 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_11_DR2_33_VC4_V11a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_11_DR2_33_VC4_V11a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_11_DR2_33_VC4_V11a_t60.bas -ss 1 [0m
     2023-05-22 16:47:55,068 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_11_DR2_33_VC4_V11a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_11_DR2_33_VC4_V11a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_11_DR2_33_VC4_V11a_t60 has 36945 rows; 37156 cols; 138884 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          18405    -4.7349339545e+04 Pr: 4472(141195) 5s
          23175    -4.7348682565e+04 Pr: 8259(1.4081e+07) 10s
          27983    -4.7343911276e+04 Pr: 5528(2.09363e+06); Du: 0(1.94289e-16) 15s
          33073    -4.7209469432e+04 Pr: 6382(1.86269e+06) 20s
          37751    -4.7172375046e+04 Pr: 7195(885014) 25s
          42523    -4.7125773198e+04 Pr: 2999(57886.8) 30s
          43968    -4.7107095441e+04 Pr: 0(0); Du: 0(1.54977e-13) 32s
          43968    -4.7107095441e+04 Pr: 0(0); Du: 0(1.54977e-13) 32s
    Model   status      : Optimal
    Simplex   iterations: 43968
    Objective value     : -4.7107095441e+04
    HiGHS run time      :         32.12
    [0m
     2023-05-22 16:47:55,069 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR2_11_DR2_33_VC5_V12a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR2_11_DR2_33_VC5_V12a_t45.bas
     2023-05-22 16:47:55,080 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_11_DR2_33_VC5_V12a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_11_DR2_33_VC5_V12a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_11_DR2_33_VC5_V12a_t45.bas -ss 1 [0m
     2023-05-22 16:48:08,141 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_11_DR2_33_VC5_V12a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_11_DR2_33_VC5_V12a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_11_DR2_33_VC5_V12a_t45 has 28051 rows; 27314 cols; 100946 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          17802    -4.3271735579e+04 Pr: 4001(406810) 5s
          25421    -4.3106109512e+04 Pr: 3343(378227) 10s
          29443    -4.2987386187e+04 Pr: 0(0); Du: 0(1.4097e-13) 12s
          29443    -4.2987386187e+04 Pr: 0(0); Du: 0(1.4097e-13) 12s
    Model   status      : Optimal
    Simplex   iterations: 29443
    Objective value     : -4.2987386187e+04
    HiGHS run time      :         12.87
    [0m
     2023-05-22 16:48:08,142 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR2_11_DR2_33_VC5_V12a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR2_11_DR2_33_VC5_V12a_t60.bas
     2023-05-22 16:48:08,152 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_11_DR2_33_VC5_V12a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_11_DR2_33_VC5_V12a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_11_DR2_33_VC5_V12a_t60.bas -ss 1 [0m
     2023-05-22 16:48:42,056 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_11_DR2_33_VC5_V12a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_11_DR2_33_VC5_V12a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_11_DR2_33_VC5_V12a_t60 has 39991 rows; 40094 cols; 150026 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          19537    -5.6871167781e+04 Pr: 5063(149109); Du: 0(1.36149e-07) 5s
          24456    -5.6851112634e+04 Pr: 6220(431920) 10s
          28685    -5.6844198825e+04 Pr: 6602(1.34928e+06); Du: 0(8.68177e-08) 15s
          33874    -5.6732731083e+04 Pr: 3877(110502); Du: 0(7.79719e-08) 20s
          38418    -5.6690413422e+04 Pr: 4858(1.4489e+06); Du: 0(5.49661e-18) 25s
          42999    -5.6586478851e+04 Pr: 3056(125499) 30s
          45543    -5.6575925931e+04 Pr: 0(0); Du: 0(2.26208e-14) 33s
          45543    -5.6575925931e+04 Pr: 0(0); Du: 0(2.26208e-14) 33s
    Model   status      : Optimal
    Simplex   iterations: 45543
    Objective value     : -5.6575925931e+04
    HiGHS run time      :         33.59
    [0m
     2023-05-22 16:48:42,057 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR2_22_DR2_22_VC3_V10a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR2_22_DR2_22_VC3_V10a_t45.bas
     2023-05-22 16:48:42,068 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_22_DR2_22_VC3_V10a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_22_DR2_22_VC3_V10a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_22_DR2_22_VC3_V10a_t45.bas -ss 1 [0m
     2023-05-22 16:48:53,258 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_22_DR2_22_VC3_V10a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_22_DR2_22_VC3_V10a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_22_DR2_22_VC3_V10a_t45 has 26234 rows; 24626 cols; 94680 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          16001    -4.5371586749e+04 Pr: 4657(809798) 5s
          24259    -4.5179771024e+04 Pr: 1714(18502.2) 10s
          25863    -4.5120476035e+04 Pr: 0(0); Du: 0(2.76809e-12) 10s
          25863    -4.5120476035e+04 Pr: 0(0); Du: 0(2.76809e-12) 10s
    Model   status      : Optimal
    Simplex   iterations: 25863
    Objective value     : -4.5120476035e+04
    HiGHS run time      :         10.99
    [0m
     2023-05-22 16:48:53,259 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR2_22_DR2_22_VC3_V10a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR2_22_DR2_22_VC3_V10a_t60.bas
     2023-05-22 16:48:53,270 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_22_DR2_22_VC3_V10a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_22_DR2_22_VC3_V10a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_22_DR2_22_VC3_V10a_t60.bas -ss 1 [0m
     2023-05-22 16:49:20,394 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_22_DR2_22_VC3_V10a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_22_DR2_22_VC3_V10a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_22_DR2_22_VC3_V10a_t60 has 37424 rows; 35936 cols; 139860 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          17726    -5.9571927951e+04 Pr: 4726(82139.3); Du: 0(1.21191e-09) 5s
          22669    -5.9569591884e+04 Pr: 7313(5.20152e+06) 10s
          27738    -5.9564808959e+04 Pr: 6424(2.01726e+06); Du: 0(8.56095e-08) 15s
          33070    -5.9429557245e+04 Pr: 4836(1.45792e+06) 20s
          38417    -5.9372822305e+04 Pr: 3587(223372); Du: 0(7.04054e-18) 25s
          39855    -5.9325360643e+04 Pr: 0(0); Du: 0(2.82982e-12) 26s
          39855    -5.9325360643e+04 Pr: 0(0); Du: 0(2.82982e-12) 26s
    Model   status      : Optimal
    Simplex   iterations: 39855
    Objective value     : -5.9325360643e+04
    HiGHS run time      :         26.83
    [0m
     2023-05-22 16:49:20,395 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR2_22_DR3_333_VC4_V14a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR2_22_DR3_333_VC4_V14a_t45.bas
     2023-05-22 16:49:20,406 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_22_DR3_333_VC4_V14a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V14a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V14a_t45.bas -ss 1 [0m
     2023-05-22 16:50:22,581 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V14a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V14a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_22_DR3_333_VC4_V14a_t45 has 60102 rows; 58969 cols; 229325 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          20644    -4.9500750314e+04 Pr: 5931(70269.1) 5s
          25076    -4.9499126709e+04 Pr: 8961(219826); Du: 0(7.90023e-09) 10s
          28813    -4.9498709844e+04 Pr: 9221(579210); Du: 0(6.88224e-08) 15s
          31941    -4.9498412591e+04 Pr: 7774(211624) 20s
          34592    -4.9479844784e+04 Pr: 10386(738973) 26s
          37595    -4.9477615514e+04 Pr: 10764(2.80734e+06) 31s
          40699    -4.9475860242e+04 Pr: 9054(1.0961e+07) 36s
          44123    -4.9440911271e+04 Pr: 6585(1.2543e+06); Du: 0(2.65938e-08) 42s
          47540    -4.9303666451e+04 Pr: 6959(343609) 47s
          50966    -4.9212767590e+04 Pr: 9278(7.86029e+06) 52s
          54138    -4.9143417604e+04 Pr: 6296(214418) 57s
          56691    -4.9107398046e+04 Pr: 0(0); Du: 0(6.71685e-14) 61s
          56691    -4.9107398046e+04 Pr: 0(0); Du: 0(6.71685e-14) 61s
    Model   status      : Optimal
    Simplex   iterations: 56691
    Objective value     : -4.9107398046e+04
    HiGHS run time      :         61.73
    [0m
     2023-05-22 16:50:22,581 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR2_22_DR3_333_VC4_V14a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR2_22_DR3_333_VC4_V14a_t60.bas
     2023-05-22 16:50:22,592 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_22_DR3_333_VC4_V14a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V14a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V14a_t60.bas -ss 1 [0m
     2023-05-22 16:53:32,308 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V14a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V14a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_22_DR3_333_VC4_V14a_t60 has 89363 rows; 90315 cols; 357660 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          22877    -6.3791202002e+04 Pr: 12006(938993); Du: 0(6.1661e-08) 5s
          26622    -6.3786174703e+04 Pr: 12428(1.17707e+06) 10s
          29636    -6.3783000108e+04 Pr: 13327(1.03757e+06); Du: 0(1.72261e-08) 16s
          31980    -6.3781882852e+04 Pr: 14858(1.29228e+06) 22s
          34359    -6.3781624924e+04 Pr: 15343(1.72518e+06) 27s
          36466    -6.3781441317e+04 Pr: 16343(6.21736e+06); Du: 0(4.59989e-08) 33s
          39109    -6.3781018557e+04 Pr: 17371(7.12982e+06) 38s
          41247    -6.3780842600e+04 Pr: 17144(1.67855e+06); Du: 0(6.34899e-08) 44s
          43136    -6.3780739247e+04 Pr: 14949(594719); Du: 0(1.23503e-07) 49s
          44878    -6.3780637528e+04 Pr: 20165(4.56582e+06); Du: 0(6.34899e-08) 55s
          46678    -6.3780453732e+04 Pr: 12778(627814); Du: 0(1.70064e-07) 60s
          48573    -6.3780037203e+04 Pr: 15708(1.25266e+07); Du: 0(6.34899e-08) 65s
          50593    -6.3779446296e+04 Pr: 15118(4.29321e+06); Du: 0(1.37308e-07) 71s
          52585    -6.3778694987e+04 Pr: 15887(9.23175e+06); Du: 0(1.44561e-07) 76s
          54526    -6.3777979993e+04 Pr: 14316(2.3875e+07); Du: 0(6.78281e-08) 82s
          56537    -6.3777296324e+04 Pr: 14546(1.32706e+07); Du: 0(9.05404e-08) 87s
          58652    -6.3776271435e+04 Pr: 12503(1.30661e+07); Du: 0(6.34899e-08) 93s
          60680    -6.3774917000e+04 Pr: 14414(6.02709e+07); Du: 0(6.34899e-08) 98s
          62706    -6.3773375602e+04 Pr: 11537(3.95072e+06); Du: 0(1.20474e-07) 104s
          64763    -6.3772431572e+04 Pr: 11837(1.26673e+07); Du: 0(6.34899e-08) 109s
          66912    -6.3769455480e+04 Pr: 13185(5.26994e+07); Du: 0(2.05328e-07) 114s
          69211    -6.3766114900e+04 Pr: 12062(1.65556e+08); Du: 0(1.42097e-07) 119s
          71370    -6.3762464951e+04 Pr: 12044(5.48777e+07); Du: 0(2.09144e-07) 125s
          73433    -6.3681655416e+04 Pr: 12910(1.59374e+07); Du: 0(1.23427e-07) 130s
          75693    -6.3621786622e+04 Pr: 11047(2.48986e+06); Du: 0(6.34899e-08) 136s
          77888    -6.3592183590e+04 Pr: 13889(5.26434e+06); Du: 0(2.05854e-07) 142s
          80095    -6.3569684048e+04 Pr: 12170(3.04973e+06); Du: 0(6.34899e-08) 148s
          82066    -6.3553818794e+04 Pr: 12078(1.00088e+06); Du: 0(1.63501e-07) 153s
          83951    -6.3550999325e+04 Pr: 16139(7.02069e+06); Du: 0(6.34899e-08) 159s
          85727    -6.3536572674e+04 Pr: 13634(2.27319e+06); Du: 0(6.34899e-08) 164s
          87781    -6.3528829605e+04 Pr: 16628(1.0338e+08) 169s
          90229    -6.3465732037e+04 Pr: 10540(4.80112e+06) 175s
          92077    -6.3419892260e+04 Pr: 10934(1.35636e+06) 180s
          93779    -6.3413366670e+04 Pr: 3544(35317.2); Du: 0(9.90566e-08) 185s
          95142    -6.3405515007e+04 Pr: 0(0); Du: 0(4.85936e-12) 189s
          95142    -6.3405515007e+04 Pr: 0(0); Du: 0(4.85936e-12) 189s
    Model   status      : Optimal
    Simplex   iterations: 95142
    Objective value     : -6.3405515007e+04
    HiGHS run time      :        189.02
    [0m
     2023-05-22 16:53:32,308 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR2_22_DR3_333_VC4_V17a_t45.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR2_22_DR3_333_VC4_V17a_t45.bas
     2023-05-22 16:53:32,318 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_22_DR3_333_VC4_V17a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V17a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V17a_t45.bas -ss 1 [0m
     2023-05-22 16:55:07,352 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V17a_t45.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V17a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_22_DR3_333_VC4_V17a_t45 has 73152 rows; 71610 cols; 280017 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          21908    -5.6091735645e+04 Pr: 8825(2.81561e+06); Du: 0(3.85637e-08) 5s
          26399    -5.6088387607e+04 Pr: 11226(9.56387e+06); Du: 0(4.91905e-15) 10s
          29688    -5.6087377064e+04 Pr: 12968(7.26698e+06); Du: 0(6.99971e-08) 15s
          32709    -5.6086915123e+04 Pr: 10524(2.162e+06); Du: 0(2.45347e-09) 20s
          35574    -5.6069390857e+04 Pr: 12419(3.67228e+06); Du: 0(8.49748e-08) 26s
          38350    -5.6061715773e+04 Pr: 14278(6.62699e+06) 31s
          40828    -5.6060891179e+04 Pr: 12525(5.64804e+06) 37s
          43177    -5.6060102493e+04 Pr: 8268(793021) 42s
          45538    -5.6058734332e+04 Pr: 10737(7.09605e+06) 47s
          48045    -5.6055463932e+04 Pr: 9319(9.78845e+06); Du: 0(1.78399e-07) 52s
          50798    -5.6040597794e+04 Pr: 8080(6.59947e+07); Du: 0(1.29923e-07) 57s
          53730    -5.5881932950e+04 Pr: 9150(509945); Du: 0(9.80645e-08) 62s
          56253    -5.5846128998e+04 Pr: 11547(8.35875e+06); Du: 0(9.25116e-09) 67s
          59084    -5.5806647297e+04 Pr: 6642(509429) 73s
          61868    -5.5707990947e+04 Pr: 12059(3.57905e+07); Du: 0(7.38651e-08) 78s
          64731    -5.5679250714e+04 Pr: 6732(148557) 84s
          67333    -5.5639438130e+04 Pr: 3969(50509.8) 89s
          69604    -5.5615665318e+04 Pr: 0(0); Du: 0(9.38277e-14) 94s
          69604    -5.5615665318e+04 Pr: 0(0); Du: 0(9.38277e-14) 94s
    Model   status      : Optimal
    Simplex   iterations: 69604
    Objective value     : -5.5615665318e+04
    HiGHS run time      :         94.49
    [0m
     2023-05-22 16:55:07,352 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
    skip: /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-no-bas/LR2_22_DR3_333_VC4_V17a_t60.log /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-basis/LR2_22_DR3_333_VC4_V17a_t60.bas
     2023-05-22 16:55:07,362 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_22_DR3_333_VC4_V17a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V17a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V17a_t60.bas -ss 1 [0m
     2023-05-22 17:00:05,077 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V17a_t60.bas/home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/opt-from-highs-ca-init-bas-m1/LR2_22_DR3_333_VC4_V17a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_22_DR3_333_VC4_V17a_t60 has 108566 rows; 109514 cols; 435517 nonzeros
    Solving LP without presolve or with basis
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0     0.0000000000e+00 Ph1: 0(0) 0s
          26016    -7.3319956010e+04 Pr: 10244(393865); Du: 0(9.70737e-08) 5s
          30836    -7.3313836225e+04 Pr: 13829(2.11348e+06); Du: 0(1.19167e-07) 11s
          33340    -7.3312630031e+04 Pr: 13480(2.51562e+06); Du: 0(9.33049e-08) 16s
          35721    -7.3310933845e+04 Pr: 17079(8.39734e+06); Du: 0(8.30278e-08) 22s
          38184    -7.3308777189e+04 Pr: 16778(2.74561e+06); Du: 0(9.71802e-08) 28s
          39834    -7.3308702753e+04 Pr: 16819(1.62853e+06); Du: 0(8.32329e-09) 33s
          41980    -7.3307873740e+04 Pr: 18361(4.51643e+06) 39s
          43809    -7.3307247229e+04 Pr: 19777(3.42901e+06); Du: 0(2.92042e-08) 44s
          45526    -7.3307173302e+04 Pr: 16870(1.03713e+06); Du: 0(2.66985e-08) 49s
          47231    -7.3307033314e+04 Pr: 20003(7.85549e+06); Du: 0(1.24019e-07) 54s
          48880    -7.3306891255e+04 Pr: 19794(6.07526e+06); Du: 0(1.98425e-07) 60s
          50374    -7.3306816879e+04 Pr: 18815(1.98206e+07); Du: 0(1.09375e-07) 66s
          52815    -7.3295239323e+04 Pr: 16768(1.56942e+06) 71s
          54281    -7.3293586821e+04 Pr: 24847(3.11325e+07); Du: 0(5.19425e-08) 77s
          56392    -7.3287706501e+04 Pr: 25228(2.6223e+07) 82s
          58371    -7.3287477432e+04 Pr: 22016(6.49279e+06) 89s
          59958    -7.3283724304e+04 Pr: 20295(3.68272e+06); Du: 0(4.94282e-08) 95s
          61525    -7.3282746990e+04 Pr: 21867(4.13735e+07) 100s
          63001    -7.3268474929e+04 Pr: 17595(2.36635e+06); Du: 0(6.11516e-08) 106s
          64600    -7.3264109830e+04 Pr: 19995(2.842e+07); Du: 0(1.36716e-07) 111s
          66127    -7.3201112218e+04 Pr: 18983(4.33653e+06); Du: 0(5.58066e-10) 117s
          67503    -7.3188880370e+04 Pr: 14489(767035); Du: 0(7.49591e-08) 122s
          68883    -7.3188057117e+04 Pr: 19465(4.65925e+07); Du: 0(4.99054e-08) 127s
          70430    -7.3186197802e+04 Pr: 16467(6.47538e+06) 132s
          72272    -7.3179733084e+04 Pr: 18705(3.25434e+07); Du: 0(1.40895e-07) 138s
          74132    -7.3175171210e+04 Pr: 15611(2.01704e+06) 144s
          75776    -7.3174013926e+04 Pr: 20799(5.41105e+07); Du: 0(1.13787e-07) 149s
          77683    -7.3161531688e+04 Pr: 18712(1.03028e+07); Du: 0(6.52006e-09) 155s
          79380    -7.3136840844e+04 Pr: 16501(1.42085e+07); Du: 0(3.90027e-08) 160s
          81205    -7.3118614131e+04 Pr: 16623(1.10701e+07); Du: 0(8.91401e-08) 166s
          83266    -7.3101955200e+04 Pr: 17416(3.68274e+08); Du: 0(6.04881e-09) 172s
          85133    -7.3099892312e+04 Pr: 15838(1.59075e+07); Du: 0(1.30651e-07) 177s
          87121    -7.3096847926e+04 Pr: 12572(5.16711e+06); Du: 0(9.05865e-08) 183s
          88778    -7.3094617406e+04 Pr: 13641(4.10129e+06); Du: 0(3.12018e-08) 188s
          90684    -7.3079920582e+04 Pr: 10984(5.07536e+06) 194s
          92673    -7.3069673676e+04 Pr: 11539(2.54102e+06) 200s
          94755    -7.3066372525e+04 Pr: 9878(850202) 206s
          96491    -7.3059673491e+04 Pr: 12485(4.00369e+06) 211s
          98015    -7.3055547436e+04 Pr: 10165(559972); Du: 0(3.24337e-08) 216s
          99518    -7.3048039113e+04 Pr: 14404(2.5583e+06) 222s
         101294    -7.3031016944e+04 Pr: 13929(3.00826e+07) 228s
         102995    -7.3012027035e+04 Pr: 13714(2.0329e+06) 233s
         104825    -7.2984381356e+04 Pr: 12353(2.18995e+06) 239s
         106590    -7.2962231971e+04 Pr: 16205(2.14211e+07) 245s
         108058    -7.2935121637e+04 Pr: 17880(1.22518e+07) 251s
         109576    -7.2916792011e+04 Pr: 9835(645141) 256s
         111034    -7.2890081483e+04 Pr: 10001(1.46685e+06) 261s
         112823    -7.2879208434e+04 Pr: 11938(275964) 268s
         114332    -7.2873119309e+04 Pr: 6755(543097) 273s
         115812    -7.2867943701e+04 Pr: 9121(470031) 279s
         117298    -7.2860883062e+04 Pr: 12607(2.28247e+06) 285s
         118910    -7.2848103841e+04 Pr: 4948(213046) 291s
         120307    -7.2838271391e+04 Pr: 0(0) 296s
         120395    -7.2838351902e+04 Pr: 0(0); Du: 0(1.28978e-10) 296s
         120395    -7.2838351902e+04 Pr: 0(0); Du: 0(1.28978e-10) 296s
    Model   status      : Optimal
    Simplex   iterations: 120395
    Objective value     : -7.2838351902e+04
    HiGHS run time      :        296.83
    [0m
     2023-05-22 17:00:05,078 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-ca-bas-m1[0m
     2023-05-22 17:00:05,089 utils.py [line:62] [32mINFO[0m Script ended successfully. Total time taken: time 868.054[0m
    ------------------------------------------------------------
    ------------------------------
     2023-05-22 17:00:05,354 utils.py [line:83] [32mINFO[0m cmd:python scripts/cvt_to_pkl.py --dev 0 --exp_nm tmp --opt adam --lr 0.001 --epochs 30 --arch GCN_FC(8,8,hids=128) --seed 0 --num_workers 8 --batch_size 327680 --load_from None --dataset mirp --solver_prefix highs- --data_prefix /home/xinglu/Downloads/lp-gnn.release/lp-dataset/ --exist_ok 1 --edge_num_thresh 12000000.0 --verbose 1 --skip_exist 1 --chunk 0/1 --loss balanced --inference_manager InferenceManager(0,) --lp_method 1 --gW None --split val --log_prefix /home/xinglu/Downloads/lp-gnn.release/runs/ --fp16 0  [0m
     2023-05-22 17:00:06,700 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 17:00:06,701 utils.py [line:103] [32mINFO[0m cmd is pwd[0m
     2023-05-22 17:00:06,710 utils.py [line:117] [32mINFO[0m stdout | /home/xinglu/Downloads/lp-gnn.release
    [0m
    proced args timer : <utils.Timer object at 0x7f5dc2fddbe0> 
    root_path : /home/xinglu/Downloads/lp-gnn.release 
    home_path : /home/xinglu 
    prefetch_factor : 4 
    ori_args : {'dev': 0, 'exp_nm': 'tmp', 'opt': 'adam', 'lr': 0.001, 'epochs': 30, 'arch': 'GCN_FC(8,8,hids=128)', 'seed': 0, 'num_workers': 8, 'batch_size': 327680, 'load_from': 'None', 'dataset': 'mirp', 'solver_prefix': 'highs-', 'data_prefix': '/home/xinglu/Downloads/lp-gnn.release/lp-dataset/', 'exist_ok': 1, 'edge_num_thresh': 12000000.0, 'verbose': 1, 'skip_exist': 1, 'f': None, 'init_method': None, 'chunk': '0/1', 'loss': 'balanced', 'inference_manager': 'InferenceManager(0,)', 'lp_method': '1', 'gW': None, 'split': 'val', 'log_prefix': '/home/xinglu/Downloads/lp-gnn.release/runs/', 'fp16': 0} 
    dev : cpu 
    exp_nm : tmp 
    opt : adam 
    lr : 0.001 
    epochs : 30 
    arch : GCN_FC(8,8,hids=128) 
    seed : 0 
    num_workers : 8 
    batch_size : 327680 
    load_from : None 
    dataset : mirp 
    solver_prefix : highs- 
    data_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset/ 
    exist_ok : 1 
    edge_num_thresh : 12000000.0 
    verbose : 1 
    skip_exist : 1 
    f : None 
    init_method : None 
    chunk : 0/1 
    loss : balanced 
    inference_manager : InferenceManager(0,) 
    lp_method : 1 
    gW : None 
    split : val 
    log_prefix : /home/xinglu/Downloads/lp-gnn.release/runs/ 
    fp16 : 0 
    dataset_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp 
    dataset_processed_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-inp_tgt/ 
    log_dir : /home/xinglu/Downloads/lp-gnn.release/runs/tmp/ 
    
     2023-05-22 17:00:06,713 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 17:00:06,713 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-inp_tgt/raw/[0m
     2023-05-22 17:00:06,764 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_3_VC1_V7a_t45.mps[0m
     2023-05-22 17:00:06,765 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V8a_t45.mps[0m
     2023-05-22 17:00:06,764 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_3_VC1_V7a_t60.mps[0m
     2023-05-22 17:00:06,765 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V8a_t60.mps[0m
     2023-05-22 17:00:06,765 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V9a_t60.mps[0m
     2023-05-22 17:00:06,765 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V9a_t45.mps[0m
     2023-05-22 17:00:06,766 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_2_DR1_3_VC2_V6a_t60.mps[0m
     2023-05-22 17:00:06,767 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_2_DR1_3_VC3_V8a_t45.mps[0m
     2023-05-22 17:00:06,768 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_2_DR1_3_VC3_V8a_t60.mps[0m
     2023-05-22 17:00:06,768 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V11a_t45.mps[0m
     2023-05-22 17:00:06,768 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V11a_t60.mps[0m
     2023-05-22 17:00:06,768 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V12a_t45.mps[0m
     2023-05-22 17:00:06,769 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V12a_t60.mps[0m
     2023-05-22 17:00:06,769 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V12b_t45.mps[0m
     2023-05-22 17:00:06,769 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_1_DR1_4_VC3_V12b_t60.mps[0m
     2023-05-22 17:00:06,769 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_11_DR2_22_VC3_V6a_t45.mps[0m
     2023-05-22 17:00:06,769 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_11_DR2_22_VC3_V6a_t60.mps[0m
     2023-05-22 17:00:06,770 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_11_DR2_33_VC4_V11a_t60.mps[0m
     2023-05-22 17:00:06,771 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_11_DR2_33_VC5_V12a_t45.mps[0m
     2023-05-22 17:00:06,771 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_11_DR2_33_VC5_V12a_t60.mps[0m
     2023-05-22 17:00:06,771 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_22_DR2_22_VC3_V10a_t45.mps[0m
     2023-05-22 17:00:06,772 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_22_DR2_22_VC3_V10a_t60.mps[0m
     2023-05-22 17:00:06,772 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_22_DR3_333_VC4_V14a_t45.mps[0m
     2023-05-22 17:00:06,772 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_22_DR3_333_VC4_V14a_t60.mps[0m
     2023-05-22 17:00:06,772 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_22_DR3_333_VC4_V17a_t45.mps[0m
     2023-05-22 17:00:06,772 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_22_DR3_333_VC4_V17a_t60.mps[0m
     2023-05-22 17:00:06,773 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR1_2_DR1_3_VC2_V6a_t45.mps[0m
     2023-05-22 17:00:06,770 cvt_to_pkl.py [line:221] [32mINFO[0m proc /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps/LR2_11_DR2_33_VC4_V11a_t45.mps[0m
     2023-05-22 17:00:06,816 utils.py [line:62] [32mINFO[0m Script ended successfully. Total time taken: time 0.115[0m
    ------------------------------------------------------------
    ------------------------------
     2023-05-22 17:00:07,053 utils.py [line:83] [32mINFO[0m cmd:python dataset.py --dev 0 --exp_nm tmp --opt adam --lr 0.001 --epochs 30 --arch GCN_FC(8,8,hids=128) --seed 0 --num_workers 8 --batch_size 327680 --load_from None --dataset mirp --solver_prefix highs- --data_prefix /home/xinglu/Downloads/lp-gnn.release/lp-dataset/ --exist_ok 1 --edge_num_thresh 12000000.0 --verbose 1 --skip_exist 1 --chunk 0/1 --loss balanced --inference_manager InferenceManager(0,) --lp_method 1 --gW None --split val --log_prefix /home/xinglu/Downloads/lp-gnn.release/runs/ --fp16 0  --skip_exist 0 [0m
     2023-05-22 17:00:08,361 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 17:00:08,362 utils.py [line:103] [32mINFO[0m cmd is pwd[0m
     2023-05-22 17:00:08,373 utils.py [line:117] [32mINFO[0m stdout | /home/xinglu/Downloads/lp-gnn.release
    [0m
    proced args timer : <utils.Timer object at 0x7f34f99a4df0> 
    root_path : /home/xinglu/Downloads/lp-gnn.release 
    home_path : /home/xinglu 
    prefetch_factor : 4 
    ori_args : {'dev': 0, 'exp_nm': 'tmp', 'opt': 'adam', 'lr': 0.001, 'epochs': 30, 'arch': 'GCN_FC(8,8,hids=128)', 'seed': 0, 'num_workers': 8, 'batch_size': 327680, 'load_from': 'None', 'dataset': 'mirp', 'solver_prefix': 'highs-', 'data_prefix': '/home/xinglu/Downloads/lp-gnn.release/lp-dataset/', 'exist_ok': 1, 'edge_num_thresh': 12000000.0, 'verbose': 1, 'skip_exist': 0, 'f': None, 'init_method': None, 'chunk': '0/1', 'loss': 'balanced', 'inference_manager': 'InferenceManager(0,)', 'lp_method': '1', 'gW': None, 'split': 'val', 'log_prefix': '/home/xinglu/Downloads/lp-gnn.release/runs/', 'fp16': 0} 
    dev : cpu 
    exp_nm : tmp 
    opt : adam 
    lr : 0.001 
    epochs : 30 
    arch : GCN_FC(8,8,hids=128) 
    seed : 0 
    num_workers : 8 
    batch_size : 327680 
    load_from : None 
    dataset : mirp 
    solver_prefix : highs- 
    data_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset/ 
    exist_ok : 1 
    edge_num_thresh : 12000000.0 
    verbose : 1 
    skip_exist : 0 
    f : None 
    init_method : None 
    chunk : 0/1 
    loss : balanced 
    inference_manager : InferenceManager(0,) 
    lp_method : 1 
    gW : None 
    split : val 
    log_prefix : /home/xinglu/Downloads/lp-gnn.release/runs/ 
    fp16 : 0 
    dataset_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp 
    dataset_processed_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-inp_tgt/ 
    log_dir : /home/xinglu/Downloads/lp-gnn.release/runs/tmp/ 
    
     2023-05-22 17:00:08,376 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 17:00:08,376 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-inp_tgt//processed/[0m
    28it [00:00, 100980.66it/s]
     2023-05-22 17:00:08,391 dataset.py [line:126] [32mINFO[0m err will recache, recahche[0m
    /home/xinglu/anaconda3/envs/p2/lib/python3.9/site-packages/torch/utils/data/dataloader.py:478: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(_create_warning_msg(
    28it [00:01, 21.72it/s]
     2023-05-22 17:00:10,081 utils.py [line:271] [32mINFO[0m split into 19 train 9 val, seed: 0[0m
     2023-05-22 17:00:10,082 utils.py [line:62] [32mINFO[0m Script ended successfully. Total time taken: time 1.721[0m
    ------------------------------------------------------------
     2023-05-22 17:00:10,333 utils.py [line:62] [32mINFO[0m Script ended successfully. Total time taken: time 874.613[0m
    ------------------------------------------------------------


## Step 2: train GNN model

We can use `run_train_test.py` (the end-to-end script) for train and test, but we will do it step by step. 

`train.py` will save the GNN model to exp_folder/mdl.pth. It spends a long time for training. For demonstration, `mdl.pth` is uploaded under `runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3`. 


```python
## skip this if do not want to training
# !python train.py --arch 'GCN_FC(8,8,hids=1024,depth=3)' --epochs 800 --loss balanced --exp_nm a_new_exp_folder --dataset mirp
```

## Step 3: evaluate GNN model and its predicted basis


```python
## predict the basis with GNN model, we will use the uploaded checkpoint.
!python scripts/pred_basis.py --arch 'GCN_FC(8,8,hids=1024,depth=3)' --exp_nm $exp_nm --load_from runs/$exp_nm/mdl.pth --dataset mirp
```

     2023-05-22 17:00:12,658 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 17:00:12,658 utils.py [line:103] [32mINFO[0m cmd is pwd[0m
     2023-05-22 17:00:12,668 utils.py [line:117] [32mINFO[0m stdout | /home/xinglu/Downloads/lp-gnn.release
    [0m
    proced args timer : <utils.Timer object at 0x7fb6a53809d0> 
    root_path : /home/xinglu/Downloads/lp-gnn.release 
    home_path : /home/xinglu 
    prefetch_factor : 4 
    ori_args : {'dev': 0, 'exp_nm': 'highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3', 'opt': 'adam', 'lr': 0.001, 'epochs': 30, 'arch': 'GCN_FC(8,8,hids=1024,depth=3)', 'seed': 0, 'num_workers': 8, 'batch_size': 327680, 'load_from': 'runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3/mdl.pth', 'dataset': 'mirp', 'solver_prefix': 'highs-', 'data_prefix': '/home/xinglu/Downloads/lp-gnn.release/lp-dataset/', 'exist_ok': 1, 'edge_num_thresh': 12000000.0, 'verbose': 1, 'skip_exist': 0, 'f': None, 'init_method': None, 'chunk': None, 'loss': 'balanced', 'inference_manager': 'InferenceManager(0,)', 'lp_method': '1', 'gW': None, 'split': 'val', 'log_prefix': '/home/xinglu/Downloads/lp-gnn.release/runs/', 'fp16': 0} 
    dev : cpu 
    exp_nm : highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3 
    opt : adam 
    lr : 0.001 
    epochs : 30 
    arch : GCN_FC(8,8,hids=1024,depth=3) 
    seed : 0 
    num_workers : 8 
    batch_size : 327680 
    load_from : runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3/mdl.pth 
    dataset : mirp 
    solver_prefix : highs- 
    data_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset/ 
    exist_ok : 1 
    edge_num_thresh : 12000000.0 
    verbose : 1 
    skip_exist : 0 
    f : None 
    init_method : None 
    chunk : None 
    loss : balanced 
    inference_manager : InferenceManager(0,) 
    lp_method : 1 
    gW : None 
    split : val 
    log_prefix : /home/xinglu/Downloads/lp-gnn.release/runs/ 
    fp16 : 0 
    dataset_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp 
    dataset_processed_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-inp_tgt/ 
    log_dir : /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3/ 
    
     2023-05-22 17:00:12,671 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 17:00:12,792 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/[0m
     2023-05-22 17:00:12,809 utils.py [line:271] [32mINFO[0m split into 19 train 9 val, seed: 0[0m
    /home/xinglu/anaconda3/envs/p2/lib/python3.9/site-packages/torch/utils/data/dataloader.py:478: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(_create_warning_msg(
      0%|                                                     | 0/9 [00:00<?, ?it/s]
      0%|                                                     | 0/1 [00:00<?, ?it/s][A
    100%|█████████████████████████████████████████████| 1/1 [00:05<00:00,  5.08s/it][A
     11%|█████                                        | 1/9 [00:05<00:43,  5.47s/it]
      0%|                                                     | 0/1 [00:00<?, ?it/s][A
    100%|█████████████████████████████████████████████| 1/1 [00:10<00:00, 10.71s/it][A
     22%|██████████                                   | 2/9 [00:16<01:00,  8.60s/it]
      0%|                                                     | 0/1 [00:00<?, ?it/s][A
    100%|█████████████████████████████████████████████| 1/1 [00:07<00:00,  7.94s/it][A
     33%|███████████████                              | 3/9 [00:24<00:49,  8.31s/it]
      0%|                                                     | 0/1 [00:00<?, ?it/s][A
    100%|█████████████████████████████████████████████| 1/1 [00:10<00:00, 10.74s/it][A
     44%|████████████████████                         | 4/9 [00:35<00:46,  9.31s/it]
      0%|                                                     | 0/1 [00:00<?, ?it/s][A
    100%|█████████████████████████████████████████████| 1/1 [00:11<00:00, 11.01s/it][A
     56%|█████████████████████████                    | 5/9 [00:46<00:39,  9.93s/it]
      0%|                                                     | 0/1 [00:00<?, ?it/s][A
    100%|█████████████████████████████████████████████| 1/1 [00:15<00:00, 15.58s/it][A
     67%|██████████████████████████████               | 6/9 [01:01<00:35, 11.89s/it]
      0%|                                                     | 0/1 [00:00<?, ?it/s][A
    100%|█████████████████████████████████████████████| 1/1 [00:22<00:00, 22.47s/it][A
     78%|███████████████████████████████████          | 7/9 [01:24<00:30, 15.41s/it]
      0%|                                                     | 0/1 [00:00<?, ?it/s][A
    100%|█████████████████████████████████████████████| 1/1 [00:24<00:00, 24.71s/it][A
     89%|████████████████████████████████████████     | 8/9 [01:49<00:18, 18.44s/it]
      0%|                                                     | 0/1 [00:00<?, ?it/s][A
    100%|█████████████████████████████████████████████| 1/1 [00:20<00:00, 20.80s/it][A
    100%|█████████████████████████████████████████████| 9/9 [02:10<00:00, 14.47s/it]
      0%|                                                     | 0/9 [00:00<?, ?it/s]/home/xinglu/anaconda3/envs/p2/lib/python3.9/site-packages/torch/utils/data/dataloader.py:478: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(_create_warning_msg(
    100%|█████████████████████████████████████████████| 9/9 [02:09<00:00, 14.36s/it]
    dump  /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//time.h5
     2023-05-22 17:04:32,642 utils.py [line:62] [32mINFO[0m Script ended successfully. Total time taken: time 259.984[0m
    ------------------------------------------------------------



```python
## run solver with predicted initial basis 
!python scripts/run_solver_from_basis.py --num_workers 1 --lp_method 1 --dataset mirp --exp_nm $exp_nm 
## report the final performance 
!python scripts/extract_time.py --exp_nm $exp_nm 
```

     2023-05-22 17:04:35,029 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 17:04:35,030 utils.py [line:103] [32mINFO[0m cmd is pwd[0m
     2023-05-22 17:04:35,040 utils.py [line:117] [32mINFO[0m stdout | /home/xinglu/Downloads/lp-gnn.release
    [0m
    proced args timer : <utils.Timer object at 0x7f5d9c7d6fa0> 
    root_path : /home/xinglu/Downloads/lp-gnn.release 
    home_path : /home/xinglu 
    prefetch_factor : 4 
    ori_args : {'dev': 0, 'exp_nm': 'highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3', 'opt': 'adam', 'lr': 0.001, 'epochs': 30, 'arch': 'GCN_FC(8,8,hids=128)', 'seed': 0, 'num_workers': 1, 'batch_size': 327680, 'load_from': 'None', 'dataset': 'mirp', 'solver_prefix': 'highs-', 'data_prefix': '/home/xinglu/Downloads/lp-gnn.release/lp-dataset/', 'exist_ok': 1, 'edge_num_thresh': 12000000.0, 'verbose': 1, 'skip_exist': 0, 'f': None, 'init_method': None, 'chunk': None, 'loss': 'balanced', 'inference_manager': 'InferenceManager(0,)', 'lp_method': '1', 'gW': None, 'split': 'val', 'log_prefix': '/home/xinglu/Downloads/lp-gnn.release/runs/', 'fp16': 0} 
    dev : cpu 
    exp_nm : highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3 
    opt : adam 
    lr : 0.001 
    epochs : 30 
    arch : GCN_FC(8,8,hids=128) 
    seed : 0 
    num_workers : 1 
    batch_size : 327680 
    load_from : None 
    dataset : mirp 
    solver_prefix : highs- 
    data_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset/ 
    exist_ok : 1 
    edge_num_thresh : 12000000.0 
    verbose : 1 
    skip_exist : 0 
    f : None 
    init_method : None 
    chunk : None 
    loss : balanced 
    inference_manager : InferenceManager(0,) 
    lp_method : 1 
    gW : None 
    split : val 
    log_prefix : /home/xinglu/Downloads/lp-gnn.release/runs/ 
    fp16 : 0 
    dataset_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp 
    dataset_processed_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-inp_tgt/ 
    log_dir : /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3/ 
    
     2023-05-22 17:04:35,042 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 17:04:35,043 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/gnn-bas-0[0m
     2023-05-22 17:04:35,053 utils.py [line:103] [32mINFO[0m cmd is rmdir /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps[0m
     2023-05-22 17:04:35,070 utils.py [line:119] [31mERROR[0m stderr | rmdir: failed to remove '/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps': Not a directory
     | cmd rmdir /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps[0m
     2023-05-22 17:04:35,070 utils.py [line:103] [32mINFO[0m cmd is ln -sf /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3/[0m
     2023-05-22 17:04:35,081 utils.py [line:103] [32mINFO[0m cmd is cp -r /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/highs-* /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/[0m
     2023-05-22 17:04:35,156 run_solver_from_basis.py [line:56] [32mINFO[0m limit run_from_basis to val only [0m
     2023-05-22 17:04:35,156 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps/LR1_1_DR1_3_VC1_V7a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR1_1_DR1_3_VC1_V7a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR1_1_DR1_3_VC1_V7a_t45.bas -ss 1 [0m
     2023-05-22 17:04:36,076 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR1_1_DR1_3_VC1_V7a_t45.bas/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR1_1_DR1_3_VC1_V7a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_3_VC1_V7a_t45 has 7909 rows; 8040 cols; 27719 nonzeros
    Solving LP without presolve or with basis
    HEkk::initialiseSimplexLpBasisAndFactor (None) Rank_deficiency 154: Id = -1; UpdateCount = -1
    HEkk::handleRankDeficiency:    0: Basic row of leaving variable (  98 is  column   98) is  580; Entering logical =  580 is variable 8620)
    HEkk::handleRankDeficiency:    1: Basic row of leaving variable ( 166 is  column  166) is  665; Entering logical =  665 is variable 8705)
    HEkk::handleRankDeficiency:  153: Basic row of leaving variable (1615 is  column 1615) is 4771; Entering logical = 4771 is variable 12811)
    time elapsed for factorize: 1.5812e-05s
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0    -1.4718183155e+01 Ph1: 3004(979898); Du: 528(14.7182) 0s
           5017    -2.5646623198e+04 Pr: 0(0); Du: 0(1.55171e-13) 0s
           5017    -2.5646623198e+04 Pr: 0(0); Du: 0(1.55171e-13) 0s
    Model   status      : Optimal
    Simplex   iterations: 5017
    Objective value     : -2.5646623198e+04
    HiGHS run time      :          0.86
    [0m
     2023-05-22 17:04:36,077 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/gnn-bas-0[0m
     2023-05-22 17:04:36,088 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps/LR1_1_DR1_4_VC3_V8a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR1_1_DR1_4_VC3_V8a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR1_1_DR1_4_VC3_V8a_t60.bas -ss 1 [0m
     2023-05-22 17:04:39,372 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR1_1_DR1_4_VC3_V8a_t60.bas/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR1_1_DR1_4_VC3_V8a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V8a_t60 has 15334 rows; 17541 cols; 60628 nonzeros
    Solving LP without presolve or with basis
    HEkk::initialiseSimplexLpBasisAndFactor (None) Rank_deficiency 244: Id = -1; UpdateCount = -1
    HEkk::handleRankDeficiency:    0: Basic row of leaving variable ( 222 is  column  222) is  100; Entering logical =  100 is variable 17641)
    HEkk::handleRankDeficiency:    1: Basic row of leaving variable ( 228 is  column  228) is  141; Entering logical =  141 is variable 17682)
    HEkk::handleRankDeficiency:  243: Basic row of leaving variable (2622 is  column 2622) is 9052; Entering logical = 9052 is variable 26593)
    time elapsed for factorize: 2.6159e-05s
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0    -3.0018655757e+01 Ph1: 9084(4.23589e+06); Du: 1670(30.0187) 0s
           8424    -3.1850671995e+04 Pr: 0(0); Du: 0(6.76119e-13) 3s
           8424    -3.1850671995e+04 Pr: 0(0); Du: 0(6.76119e-13) 3s
    Model   status      : Optimal
    Simplex   iterations: 8424
    Objective value     : -3.1850671995e+04
    HiGHS run time      :          3.17
    [0m
     2023-05-22 17:04:39,373 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/gnn-bas-0[0m
     2023-05-22 17:04:39,384 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps/LR1_2_DR1_3_VC2_V6a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR1_2_DR1_3_VC2_V6a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR1_2_DR1_3_VC2_V6a_t60.bas -ss 1 [0m
     2023-05-22 17:04:41,667 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR1_2_DR1_3_VC2_V6a_t60.bas/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR1_2_DR1_3_VC2_V6a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_2_DR1_3_VC2_V6a_t60 has 12876 rows; 13553 cols; 48642 nonzeros
    Solving LP without presolve or with basis
    HEkk::initialiseSimplexLpBasisAndFactor (None) Rank_deficiency 108: Id = -1; UpdateCount = -1
    HEkk::handleRankDeficiency:    0: Basic row of leaving variable (  56 is  column   56) is  120; Entering logical =  120 is variable 13673)
    HEkk::handleRankDeficiency:    1: Basic row of leaving variable ( 116 is  column  116) is  153; Entering logical =  153 is variable 13706)
    HEkk::handleRankDeficiency:  107: Basic row of leaving variable (4825 is  column 4825) is 3484; Entering logical = 3484 is variable 17037)
    time elapsed for factorize: 1.1066e-05s
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0    -6.0619572812e+00 Ph1: 931(678552); Du: 93(6.06196) 0s
           7149    -3.0588817139e+04 Pr: 0(0); Du: 0(5.91273e-13) 2s
           7149    -3.0588817139e+04 Pr: 0(0); Du: 0(5.91273e-13) 2s
    Model   status      : Optimal
    Simplex   iterations: 7149
    Objective value     : -3.0588817139e+04
    HiGHS run time      :          2.18
    [0m
     2023-05-22 17:04:41,668 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/gnn-bas-0[0m
     2023-05-22 17:04:41,678 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps/LR1_2_DR1_3_VC3_V8a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR1_2_DR1_3_VC3_V8a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR1_2_DR1_3_VC3_V8a_t60.bas -ss 1 [0m
     2023-05-22 17:04:45,441 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR1_2_DR1_3_VC3_V8a_t60.bas/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR1_2_DR1_3_VC3_V8a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_2_DR1_3_VC3_V8a_t60 has 17035 rows; 17885 cols; 64912 nonzeros
    Solving LP without presolve or with basis
    HEkk::initialiseSimplexLpBasisAndFactor (None) Rank_deficiency 183: Id = -1; UpdateCount = -1
    HEkk::handleRankDeficiency:    0: Basic row of leaving variable (  26 is  column   26) is   58; Entering logical =   58 is variable 17943)
    HEkk::handleRankDeficiency:    1: Basic row of leaving variable (  45 is  column   45) is  338; Entering logical =  338 is variable 18223)
    HEkk::handleRankDeficiency:  182: Basic row of leaving variable (5955 is  column 5955) is 9271; Entering logical = 9271 is variable 27156)
    time elapsed for factorize: 1.6255e-05s
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0    -7.7957596894e+00 Ph1: 1734(2.47856e+06); Du: 506(7.79576) 0s
           8930    -3.6754344157e+04 Pr: 0(0); Du: 0(1.9547e-12) 3s
           8930    -3.6754344157e+04 Pr: 0(0); Du: 0(1.9547e-12) 3s
    Model   status      : Optimal
    Simplex   iterations: 8930
    Objective value     : -3.6754344157e+04
    HiGHS run time      :          3.63
    [0m
     2023-05-22 17:04:45,442 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/gnn-bas-0[0m
     2023-05-22 17:04:45,453 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps/LR1_1_DR1_4_VC3_V12a_t45.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR1_1_DR1_4_VC3_V12a_t45.bas -bo /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR1_1_DR1_4_VC3_V12a_t45.bas -ss 1 [0m
     2023-05-22 17:04:48,669 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR1_1_DR1_4_VC3_V12a_t45.bas/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR1_1_DR1_4_VC3_V12a_t45.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V12a_t45 has 16037 rows; 17440 cols; 60442 nonzeros
    Solving LP without presolve or with basis
    HEkk::initialiseSimplexLpBasisAndFactor (None) Rank_deficiency 407: Id = -1; UpdateCount = -1
    HEkk::handleRankDeficiency:    0: Basic row of leaving variable ( 189 is  column  189) is  683; Entering logical =  683 is variable 18123)
    HEkk::handleRankDeficiency:    1: Basic row of leaving variable ( 231 is  column  231) is  707; Entering logical =  707 is variable 18147)
    HEkk::handleRankDeficiency:  406: Basic row of leaving variable (6231 is  column 6231) is 9583; Entering logical = 9583 is variable 27023)
    time elapsed for factorize: 5.6207e-05s
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0    -4.2311083403e+01 Ph1: 3977(349655); Du: 1069(42.3111) 0s
           9378    -3.3450001287e+04 Pr: 0(0); Du: 0(3.93026e-11) 3s
           9378    -3.3450001287e+04 Pr: 0(0); Du: 0(3.93026e-11) 3s
    Model   status      : Optimal
    Simplex   iterations: 9378
    Objective value     : -3.3450001287e+04
    HiGHS run time      :          3.09
    [0m
     2023-05-22 17:04:48,670 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/gnn-bas-0[0m
     2023-05-22 17:04:48,681 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps/LR1_1_DR1_4_VC3_V12b_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR1_1_DR1_4_VC3_V12b_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR1_1_DR1_4_VC3_V12b_t60.bas -ss 1 [0m
     2023-05-22 17:04:57,410 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR1_1_DR1_4_VC3_V12b_t60.bas/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR1_1_DR1_4_VC3_V12b_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR1_1_DR1_4_VC3_V12b_t60 has 22443 rows; 25309 cols; 88358 nonzeros
    Solving LP without presolve or with basis
    HEkk::initialiseSimplexLpBasisAndFactor (None) Rank_deficiency 412: Id = -1; UpdateCount = -1
    HEkk::handleRankDeficiency:    0: Basic row of leaving variable (  73 is  column   73) is   80; Entering logical =   80 is variable 25389)
    HEkk::handleRankDeficiency:    1: Basic row of leaving variable ( 181 is  column  181) is   92; Entering logical =   92 is variable 25401)
    HEkk::handleRankDeficiency:  411: Basic row of leaving variable (8540 is  column 8540) is 13618; Entering logical = 13618 is variable 38927)
    time elapsed for factorize: 6.0972e-05s
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0    -3.9526642980e+01 Ph1: 3794(1.14132e+06); Du: 1000(39.5266) 0s
          11191    -4.0196190563e+04 Pr: 4360(3.09733e+06); Du: 0(8.42939e-08) 5s
          16342    -4.0065391472e+04 Pr: 0(0); Du: 0(3.21965e-15) 8s
          16342    -4.0065391472e+04 Pr: 0(0); Du: 0(3.21965e-15) 8s
    Model   status      : Optimal
    Simplex   iterations: 16342
    Objective value     : -4.0065391472e+04
    HiGHS run time      :          8.56
    [0m
     2023-05-22 17:04:57,411 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/gnn-bas-0[0m
     2023-05-22 17:04:57,421 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps/LR2_11_DR2_33_VC4_V11a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR2_11_DR2_33_VC4_V11a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR2_11_DR2_33_VC4_V11a_t60.bas -ss 1 [0m
     2023-05-22 17:05:23,125 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR2_11_DR2_33_VC4_V11a_t60.bas/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR2_11_DR2_33_VC4_V11a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_11_DR2_33_VC4_V11a_t60 has 36945 rows; 37156 cols; 138884 nonzeros
    Solving LP without presolve or with basis
    HEkk::initialiseSimplexLpBasisAndFactor (None) Rank_deficiency 1016: Id = -1; UpdateCount = -1
    HEkk::handleRankDeficiency:    0: Basic row of leaving variable (  69 is  column   69) is  425; Entering logical =  425 is variable 37581)
    HEkk::handleRankDeficiency:    1: Basic row of leaving variable (  73 is  column   73) is  681; Entering logical =  681 is variable 37837)
    HEkk::handleRankDeficiency: 1015: Basic row of leaving variable (12324 is  column 12324) is 22128; Entering logical = 22128 is variable 59284)
    time elapsed for factorize: 0.000163403s
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0    -2.0312821709e+04 Ph1: 13774(2.78965e+06); Du: 1770(20312.8) 0s
          11305    -4.7348666300e+04 Pr: 5045(660689); Du: 0(8.39679e-08) 5s
          16824    -4.7343777551e+04 Pr: 4271(1.34777e+06); Du: 0(7.13449e-08) 10s
          21845    -4.7224572853e+04 Pr: 6072(1.38188e+06) 15s
          26358    -4.7164955428e+04 Pr: 5689(770685) 20s
          30817    -4.7107095441e+04 Pr: 0(0); Du: 0(5.22331e-12) 25s
          30817    -4.7107095441e+04 Pr: 0(0); Du: 0(5.22331e-12) 25s
    Model   status      : Optimal
    Simplex   iterations: 30817
    Objective value     : -4.7107095441e+04
    HiGHS run time      :         25.42
    [0m
     2023-05-22 17:05:23,126 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/gnn-bas-0[0m
     2023-05-22 17:05:23,137 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps/LR2_11_DR2_33_VC5_V12a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR2_11_DR2_33_VC5_V12a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR2_11_DR2_33_VC5_V12a_t60.bas -ss 1 [0m
     2023-05-22 17:05:54,613 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR2_11_DR2_33_VC5_V12a_t60.bas/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR2_11_DR2_33_VC5_V12a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_11_DR2_33_VC5_V12a_t60 has 39991 rows; 40094 cols; 150026 nonzeros
    Solving LP without presolve or with basis
    HEkk::initialiseSimplexLpBasisAndFactor (None) Rank_deficiency 1079: Id = -1; UpdateCount = -1
    HEkk::handleRankDeficiency:    0: Basic row of leaving variable (  59 is  column   59) is  937; Entering logical =  937 is variable 41031)
    HEkk::handleRankDeficiency:    1: Basic row of leaving variable (  69 is  column   69) is 1038; Entering logical = 1038 is variable 41132)
    HEkk::handleRankDeficiency: 1078: Basic row of leaving variable (13101 is  column 13101) is 20773; Entering logical = 20773 is variable 60867)
    time elapsed for factorize: 0.000185149s
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0    -5.6932288316e+03 Ph1: 6737(1.74036e+06); Du: 2086(5693.23) 0s
          11353    -5.6870573869e+04 Pr: 5721(473358); Du: 0(1.10891e-07) 5s
          16042    -5.6843126542e+04 Pr: 5646(272656); Du: 0(1.27706e-07) 10s
          20683    -5.6834294721e+04 Pr: 6141(2.21528e+06); Du: 0(9.14811e-08) 15s
          25720    -5.6728920264e+04 Pr: 3696(366450); Du: 0(8.98081e-08) 20s
          30351    -5.6652259770e+04 Pr: 5697(575322); Du: 0(1.26205e-07) 25s
          34733    -5.6576987750e+04 Pr: 995(38565.6) 30s
          35278    -5.6575925931e+04 Pr: 0(0); Du: 0(5.19942e-13) 31s
          35278    -5.6575925931e+04 Pr: 0(0); Du: 0(5.19942e-13) 31s
    Model   status      : Optimal
    Simplex   iterations: 35278
    Objective value     : -5.6575925931e+04
    HiGHS run time      :         31.20
    [0m
     2023-05-22 17:05:54,614 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/gnn-bas-0[0m
     2023-05-22 17:05:54,625 utils.py [line:103] [32mINFO[0m cmd is highs --model_file /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps/LR2_22_DR2_22_VC3_V10a_t60.mps --presolve off --solver simplex --random_seed 0  -bi /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR2_22_DR2_22_VC3_V10a_t60.bas -bo /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR2_22_DR2_22_VC3_V10a_t60.bas -ss 1 [0m
     2023-05-22 17:06:13,256 utils.py [line:117] [32mINFO[0m stdout | xinglu: 1/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//pred-basis/LR2_22_DR2_22_VC3_V10a_t60.bas/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//opt-from-pred-basis/LR2_22_DR2_22_VC3_V10a_t60.bas
    xinglu: ss 1
    Running HiGHS 1.3.1 [date: 2023-05-22, git hash: n/a]
    Copyright (c) 2022 ERGO-Code under MIT licence terms
    LP   LR2_22_DR2_22_VC3_V10a_t60 has 37424 rows; 35936 cols; 139860 nonzeros
    Solving LP without presolve or with basis
    HEkk::initialiseSimplexLpBasisAndFactor (None) Rank_deficiency 1404: Id = -1; UpdateCount = -1
    HEkk::handleRankDeficiency:    0: Basic row of leaving variable (   5 is  column    5) is  168; Entering logical =  168 is variable 36104)
    HEkk::handleRankDeficiency:    1: Basic row of leaving variable (   7 is  column    7) is  293; Entering logical =  293 is variable 36229)
    HEkk::handleRankDeficiency: 1403: Basic row of leaving variable (40916 is logical 4980) is 20230; Entering logical = 20230 is variable 56166)
    time elapsed for factorize: 0.000314466s
    Using EKK dual simplex solver - serial
      Iteration        Objective     Infeasibilities num(sum)
              0    -9.3598128562e+01 Ph1: 8116(1.90423e+06); Du: 2932(93.5981) 0s
          13262    -5.9554803296e+04 Pr: 7198(2.89525e+06); Du: 0(1.0617e-07) 5s
          17937    -5.9488120200e+04 Pr: 6622(2.02843e+06) 10s
          23099    -5.9409370076e+04 Pr: 4449(2.60461e+06); Du: 0(1.3035e-07) 15s
          26635    -5.9325360643e+04 Pr: 0(0); Du: 0(5.62467e-14) 18s
          26635    -5.9325360643e+04 Pr: 0(0); Du: 0(5.62467e-14) 18s
    Model   status      : Optimal
    Simplex   iterations: 26635
    Objective value     : -5.9325360643e+04
    HiGHS run time      :         18.37
    [0m
     2023-05-22 17:06:13,257 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/gnn-bas-0[0m
     2023-05-22 17:06:13,268 utils.py [line:62] [32mINFO[0m Script ended successfully. Total time taken: time 98.238[0m
    ------------------------------------------------------------
     2023-05-22 17:06:15,476 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 17:06:15,476 utils.py [line:103] [32mINFO[0m cmd is pwd[0m
     2023-05-22 17:06:15,486 utils.py [line:117] [32mINFO[0m stdout | /home/xinglu/Downloads/lp-gnn.release
    [0m
    proced args timer : <utils.Timer object at 0x7f5bdc8e69d0> 
    root_path : /home/xinglu/Downloads/lp-gnn.release 
    home_path : /home/xinglu 
    prefetch_factor : 4 
    ori_args : {'dev': 0, 'exp_nm': 'highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3', 'opt': 'adam', 'lr': 0.001, 'epochs': 30, 'arch': 'GCN_FC(8,8,hids=128)', 'seed': 0, 'num_workers': 8, 'batch_size': 327680, 'load_from': 'None', 'dataset': 'mirp', 'solver_prefix': 'highs-', 'data_prefix': '/home/xinglu/Downloads/lp-gnn.release/lp-dataset/', 'exist_ok': 1, 'edge_num_thresh': 12000000.0, 'verbose': 1, 'skip_exist': 0, 'f': None, 'init_method': None, 'chunk': None, 'loss': 'balanced', 'inference_manager': 'InferenceManager(0,)', 'lp_method': '1', 'gW': None, 'split': 'val', 'log_prefix': '/home/xinglu/Downloads/lp-gnn.release/runs/', 'fp16': 0} 
    dev : cpu 
    exp_nm : highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3 
    opt : adam 
    lr : 0.001 
    epochs : 30 
    arch : GCN_FC(8,8,hids=128) 
    seed : 0 
    num_workers : 8 
    batch_size : 327680 
    load_from : None 
    dataset : mirp 
    solver_prefix : highs- 
    data_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset/ 
    exist_ok : 1 
    edge_num_thresh : 12000000.0 
    verbose : 1 
    skip_exist : 0 
    f : None 
    init_method : None 
    chunk : None 
    loss : balanced 
    inference_manager : InferenceManager(0,) 
    lp_method : 1 
    gW : None 
    split : val 
    log_prefix : /home/xinglu/Downloads/lp-gnn.release/runs/ 
    fp16 : 0 
    dataset_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp 
    dataset_processed_prefix : /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/highs-inp_tgt/ 
    log_dir : /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3/ 
    
     2023-05-22 17:06:15,488 utils.py [line:486] [32mINFO[0m stream logging set up [0m
     2023-05-22 17:06:15,489 utils.py [line:103] [32mINFO[0m cmd is rmdir /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps[0m
     2023-05-22 17:06:15,499 utils.py [line:119] [31mERROR[0m stderr | rmdir: failed to remove '/home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps': Not a directory
     | cmd rmdir /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps[0m
     2023-05-22 17:06:15,499 utils.py [line:103] [32mINFO[0m cmd is rm /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//mps[0m
     2023-05-22 17:06:15,509 utils.py [line:103] [32mINFO[0m cmd is ln -sf /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/mps /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//[0m
     2023-05-22 17:06:15,519 utils.py [line:103] [32mINFO[0m cmd is mkdir -p /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log[0m
     2023-05-22 17:06:15,530 utils.py [line:103] [32mINFO[0m cmd is cp -r /home/xinglu/Downloads/lp-gnn.release/lp-dataset//mirp/log/* /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/[0m
     2023-05-22 17:06:15,548 utils.py [line:103] [32mINFO[0m cmd is cp -r /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/highs-ca-bas-m1 /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//log/highs-ca-bas[0m
    split                           train        val
    highs-no-bas/niter    mean  36438.158  29858.000
                          std   34997.310  16997.537
    highs-no-bas/time     mean     46.957     19.163
                          std      92.434     18.221
    highs-ca-bas/niter    mean  32220.316  26208.333
                          std   31202.713  14185.227
    highs-ca-bas/time     mean     43.713     17.359
                          std      85.872     15.633
    highs-ca-bas-m1/niter mean  32220.316  26208.333
                          std   31202.713  14185.227
    highs-ca-bas-m1/time  mean     38.603     14.238
                          std      77.849     13.156
    gnn-bas-0/niter       mean        NaN  16441.111
                          std         NaN  11471.987
    gnn-bas-0/time        mean        NaN     10.720
                          std         NaN     11.371
                                                                0
    highs-no-bas/niter     $29.9K{\scriptscriptstyle  \pm 17.0K}$
    highs-no-bas/time        $19.2{\scriptscriptstyle  \pm 18.2}$
    highs-ca-bas/niter     $26.2K{\scriptscriptstyle  \pm 14.2K}$
    highs-ca-bas/time        $17.4{\scriptscriptstyle  \pm 15.6}$
    highs-ca-bas-m1/niter  $26.2K{\scriptscriptstyle  \pm 14.2K}$
    highs-ca-bas-m1/time     $14.2{\scriptscriptstyle  \pm 13.2}$
    gnn-bas-0/niter        $16.4K{\scriptscriptstyle  \pm 11.5K}$
    gnn-bas-0/time           $10.7{\scriptscriptstyle  \pm 11.4}$
    dump  /home/xinglu/Downloads/lp-gnn.release/runs/highs--mirp-balance-ep800-archGCN_FC-8-8-hids-1024-depth-3//time.h5
     2023-05-22 17:06:15,752 utils.py [line:62] [32mINFO[0m Script ended successfully. Total time taken: time 0.276[0m
    ------------------------------------------------------------



```python
## tidy up before release the demo
# !rm -rf ./HiGHS-master/build __pycache__ 
```


```python

```
