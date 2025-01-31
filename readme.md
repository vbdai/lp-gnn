# Smart Initial Basis Selection for Linear Programs

### This repo contains the implmenetation of the paper 'Smart Initial Basis Selection for Linear Programs' published in ICML2023.

Link to arXiv paper: https://openreview.net/forum?id=Mha86sOok1

Link to Huawei's AI Gallery Notebook: [https://developer.huaweicloud.com/develop/aigallery/notebook/detail?id=58b799a0-5cfc-4c2e-8b9b-440bb2315264](https://developer.huaweicloud.com/develop/aigallery/notebook/detail?id=ce45dd10-44ce-43bb-89c8-1f3277f1132d)

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

```python
## run solver with predicted initial basis 
!python scripts/run_solver_from_basis.py --num_workers 1 --lp_method 1 --dataset mirp --exp_nm $exp_nm 
## report the final performance 
!python scripts/extract_time.py --exp_nm $exp_nm 
```

```python
## tidy up before release the demo
# !rm -rf ./HiGHS-master/build __pycache__ 
```
