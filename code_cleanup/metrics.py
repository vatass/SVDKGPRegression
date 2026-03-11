import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
import pandas as pd
import gpytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from functions import process_temporal_singletask_data
import json
import pickle
import argparse
import os
import matplotlib.pyplot as plt 
import datetime
# Define Regions ROIs
import json
from collections import OrderedDict
torch.set_default_dtype(torch.float64)
import math

def gaussian_nll_per_task(y_true, mean, var, eps=1e-9):
    var = np.maximum(var, eps)
    nll = 0.5 * (np.log(2.0 * math.pi * var) + ((y_true - mean) ** 2) / var)
    nll_per_task = np.mean(nll, axis=0)
    return nll_per_task, float(np.mean(nll_per_task))

def coverage_and_width_per_task(y_true, mean, var, z=1.96, eps=1e-9):
    std = np.sqrt(np.maximum(var, eps))
    lower = mean - z * std
    upper = mean + z * std
    within = (y_true >= lower) & (y_true <= upper)
    coverage_per_task = 100.0 * np.mean(within, axis=0)
    width_per_task = np.mean(upper - lower, axis=0)
    return coverage_per_task, width_per_task, float(np.mean(coverage_per_task)), float(np.mean(width_per_task))

def _mse_mae_per_task(actuals_list, preds_list):
    mse_per_task = []
    mae_per_task = []
    for k in range(len(actuals_list)):
        mse_per_task.append(mean_squared_error(actuals_list[k], preds_list[k]))
        mae_per_task.append(mean_absolute_error(actuals_list[k], preds_list[k]))
    return mse_per_task, mae_per_task, float(np.mean(mse_per_task)), float(np.mean(mae_per_task))