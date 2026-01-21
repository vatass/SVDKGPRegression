#!/bin/bash

python multitasksvdkgpregression.py 
python multitasksvdkgpregression.py --lambda_val 0.1
python multitasksvdkgpregression.py --lambda_val 0.5
python multitasksvdkgpregression.py --lambda_val 1.0
python multitasksvdkgpregression.py --lambda_val 2.0