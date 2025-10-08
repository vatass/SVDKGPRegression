#!/bin/bash

python svdkgpregressionmonotonicity.py --lambda_penalty 0.0
python svdkgpregressionmonotonicity.py --lambda_penalty 0.75
python svdkgpregressionmonotonicity.py --lambda_penalty 1.0
python svdkgpregressionmonotonicity.py --lambda_penalty 5.0
