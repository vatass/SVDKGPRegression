#!/bin/bash

python svdkgpregressionmonotonicity.py --lambda_penalty 0.0 --data_file "H_MUSE_Volume_47"
python svdkgpregressionmonotonicity.py --lambda_penalty 0.75 --data_file "H_MUSE_Volume_47"
python svdkgpregressionmonotonicity.py --lambda_penalty 1.0 --data_file "H_MUSE_Volume_47"
python svdkgpregressionmonotonicity.py --lambda_penalty 5.0 --data_file "H_MUSE_Volume_47"
python check_monotonicity.py --data_file "MUSE/H_MUSE_Volume_47"

python svdkgpregressionmonotonicity.py --lambda_penalty 0.0 --data_file "H_MUSE_Volume_48"
python svdkgpregressionmonotonicity.py --lambda_penalty 0.75 --data_file "H_MUSE_Volume_48"
python svdkgpregressionmonotonicity.py --lambda_penalty 1.0 --data_file "H_MUSE_Volume_48"
python svdkgpregressionmonotonicity.py --lambda_penalty 5.0 --data_file "H_MUSE_Volume_48"
python check_monotonicity.py --data_file "MUSE/H_MUSE_Volume_48"

python svdkgpregressionmonotonicity.py --lambda_penalty 0.0 --data_file "H_MUSE_Volume_51"
python svdkgpregressionmonotonicity.py --lambda_penalty 0.75 --data_file "H_MUSE_Volume_51"
python svdkgpregressionmonotonicity.py --lambda_penalty 1.0 --data_file "H_MUSE_Volume_51"
python svdkgpregressionmonotonicity.py --lambda_penalty 5.0 --data_file "H_MUSE_Volume_51"
python check_monotonicity.py --data_file "MUSE/H_MUSE_Volume_51"

python svdkgpregressionmonotonicity.py --lambda_penalty 0.0 --data_file "H_MUSE_Volume_52"
python svdkgpregressionmonotonicity.py --lambda_penalty 0.75 --data_file "H_MUSE_Volume_52"
python svdkgpregressionmonotonicity.py --lambda_penalty 1.0 --data_file "H_MUSE_Volume_52"
python svdkgpregressionmonotonicity.py --lambda_penalty 5.0 --data_file "H_MUSE_Volume_52"
python check_monotonicity.py --data_file "MUSE/H_MUSE_Volume_52"

python svdkgpregressionmonotonicity.py --lambda_penalty 0.0 --task "SPARE_AD"
python svdkgpregressionmonotonicity.py --lambda_penalty 0.75 --task "SPARE_AD"
python svdkgpregressionmonotonicity.py --lambda_penalty 1.0 --task "SPARE_AD"
python svdkgpregressionmonotonicity.py --lambda_penalty 5.0 --task "SPARE_AD"
python check_monotonicity.py --data_file "SPARE_AD"

python svdkgpregressionmonotonicity.py --lambda_penalty 0.0 --task "SPARE_BA"
python svdkgpregressionmonotonicity.py --lambda_penalty 0.75 --task "SPARE_BA"
python svdkgpregressionmonotonicity.py --lambda_penalty 1.0 --task "SPARE_BA"
python svdkgpregressionmonotonicity.py --lambda_penalty 5.0 --task "SPARE_BA"
python check_monotonicity.py --data_file "SPARE_BA"

python svdkgpregressionmonotonicity.py --lambda_penalty 0.0 --task "ADAS"
python svdkgpregressionmonotonicity.py --lambda_penalty 0.75 --task "ADAS"
python svdkgpregressionmonotonicity.py --lambda_penalty 1.0 --task "ADAS"
python svdkgpregressionmonotonicity.py --lambda_penalty 5.0 --task "ADAS"
python check_monotonicity.py --data_file "ADAS"

python svdkgpregressionmonotonicity.py --lambda_penalty 0.0 --task "MMSE"
python svdkgpregressionmonotonicity.py --lambda_penalty 0.75 --task "MMSE"
python svdkgpregressionmonotonicity.py --lambda_penalty 1.0 --task "MMSE"
python svdkgpregressionmonotonicity.py --lambda_penalty 5.0 --task "MMSE"
python check_monotonicity.py --data_file "MMSE"