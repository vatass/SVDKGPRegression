#!/bin/bash

python multitasksvdkgpregression.py --lambda_val 0.0 --mode 0 --hidden_dim 64 --points 4 --epochs 50
# python multitasksvdkgpregression.py --lambda_val 0.5 --mode 0
python multitasksvdkgpregression.py --lambda_val 1.0 --mode 0 --hidden_dim 64 --points 4 --epochs 50
# python multitasksvdkgpregression.py --lambda_val 2.0 --mode 0

python multitasksvdkgpregression.py --lambda_val 0.0 --mode 1 --hidden_dim 64 --points 4 --epochs 50
#python multitasksvdkgpregression.py --lambda_val 0.5 --mode 1
python multitasksvdkgpregression.py --lambda_val 1.0 --mode 1 --hidden_dim 64  --points 4 --epochs 50
#python multitasksvdkgpregression.py --lambda_val 2.0 --mode 1

python multitasksvdkgpregression.py --lambda_val 0.0 --mode 2 --hidden_dim 64  --points 4 --epochs 50
#python multitasksvdkgpregression.py --lambda_val 0.5 --mode 2
python multitasksvdkgpregression.py --lambda_val 1.0 --mode 2 --hidden_dim 64  --points 4 --epochs 50
#python multitasksvdkgpregression.py --lambda_val 2.0 --mode 2

python multitasksvdkgpregression.py --lambda_val 0.0 --mode 3 --hidden_dim 64  --points 4 --epochs 50
#python multitasksvdkgpregression.py --lambda_val 0.5 --mode 3
python multitasksvdkgpregression.py --lambda_val 1.0 --mode 3 --hidden_dim 64  --points 4 --epochs 50
#python multitasksvdkgpregression.py --lambda_val 2.0 --mode 3

python multitasksvdkgpregression.py --lambda_val 0.0 --mode 4 --hidden_dim 64  --points 4 --epochs 50
#python multitasksvdkgpregression.py --lambda_val 0.5 --mode 4
python multitasksvdkgpregression.py --lambda_val 1.0 --mode 4 --hidden_dim 64  --points 4 --epochs 50
#python multitasksvdkgpregression.py --lambda_val 2.0 --mode 4

python multitasksvdkgpregression.py --lambda_val 0.1 --mode 5 --hidden_dim 128 --points 4 --epochs 50
#python multitasksvdkgpregression.py --lambda_val 0.5 --mode 5
python multitasksvdkgpregression.py --lambda_val 1.0 --mode 5 --hidden_dim 128  --points 4 --epochs 50
#python multitasksvdkgpregression.py --lambda_val 2.0 --mode 5