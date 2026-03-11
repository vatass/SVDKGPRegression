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

def get_region_y_indices_and_names(mode: int, roi_to_idx: dict):
    if mode not in REGION_ROIS:
        raise ValueError(f"Mode {mode} not supported")
    
    region_name, rois = REGION_ROIS[mode]
    roi_ids = []
    for roi in rois:
        if isinstance(roi, (list, tuple)):
            iterable = roi
        else:
            iterable = [roi]
        
        for r in iterable:
            r = str(r).strip()
            if r.startswith("MUSE_Volume_"):
                r = r.split("_")[-1]
            roi_ids.append(r)

    missing = [rid for rid in roi_ids if rid not in roi_to_idx]
    print(roi_to_idx)
    if missing:
        raise KeyError(f"These ROIS are missing: {missing}")
    
    idxs = [int(roi_to_idx[rid]) for rid in roi_ids]

    task_names = [f"ROI_{rid}" for rid in roi_ids]
    
    return region_name, idxs, task_names


# Step 7: Define the select_inducing_points function
def select_inducing_points(train_x, train_subject_ids, selected_subject_ids=None, num_points_per_subject=3, device='cuda'):
    """
    Selects inducing points by sampling observations from selected subjects.

    Parameters:
    - train_x: NumPy array of shape (num_samples, input_dim)
    - train_subject_ids: List or array of subject IDs corresponding to each row in train_x
    - selected_subject_ids: List of subject IDs from which to select inducing points (optional)
    - num_points_per_subject: Number of inducing points to select per subject

    Returns:
    - inducing_points: Torch tensor of selected inducing points
    """

    # Determine the expected number of features
    expected_num_features = train_x.shape[1]

    # Create a DataFrame for easier manipulation
    data = pd.DataFrame(train_x)
    data['PTID'] = train_subject_ids

    inducing_points_list = []

    if selected_subject_ids is not None:
        # Filter data to include only selected subjects
        data_selected = data[data['PTID'].isin(selected_subject_ids)]
    else:
        data_selected = data

    # Group by subject
    grouped = data_selected.groupby('PTID')

    for subject_id, group in grouped:
        # Sort the group by temporal variable (assuming it's the last column minus one)
        temporal_col_index = expected_num_features - 1  # Adjust index if necessary
        group_sorted = group.sort_values(by=group.columns[temporal_col_index])
        num_observations = group_sorted.shape[0]

        if num_observations >= num_points_per_subject:
            # Select earliest, median, and latest time points
            indices = [0, num_observations // 2, num_observations - 1]
            selected_points = group_sorted.iloc[indices]
        else:
            # If less than num_points_per_subject observations, select all
            selected_points = group_sorted

        # Drop 'PTID' column and convert to values
        selected_values = selected_points.drop('PTID', axis=1).values

        if selected_values.size > 0:
            if selected_values.shape[1] != expected_num_features:
                print(f"Warning: Subject {subject_id} has unexpected number of features: {selected_values.shape[1]} (expected {expected_num_features}).")
                continue  # Skip this subject or handle as appropriate

            # Ensure numerical data type
            selected_values = selected_values.astype(np.float64)
            inducing_points_list.append(selected_values)
        else:
            print(f"Warning: No inducing points for subject {subject_id}.")

    if inducing_points_list:
        inducing_points_array = np.vstack(inducing_points_list)
        inducing_points = torch.tensor(inducing_points_array, dtype=torch.float64)
        inducing_points = inducing_points.to(device)
    else:
        raise ValueError("No inducing points were selected. Check your data and selection criteria.")

    return inducing_points

def load_and_preprocess_region_based_data(folder, file, train_ids, test_ids, mode: int):
    f = open('/home/cbica/Desktop/LongGPClustering/roi_to_idx.json')
    roi_to_idx = json.load(f)

    index_to_roi = {v: k for k, v in roi_to_idx.items()}
 
    datasamples = pd.read_csv('/home/cbica/Desktop/DKGP/data/subjectsamples_longclean_dl_muse_allstudies.csv')

    train_x = datasamples[datasamples['PTID'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['PTID'].isin(train_ids)]['Y']
    test_x  = datasamples[datasamples['PTID'].isin(test_ids)]['X']
    test_y  = datasamples[datasamples['PTID'].isin(test_ids)]['Y']

    corresponding_train_ids = datasamples[datasamples['PTID'].isin(train_ids)]['PTID'].tolist()
    corresponding_test_ids  = datasamples[datasamples['PTID'].isin(test_ids)]['PTID'].tolist()

    train_x, train_y, test_x, test_y = process_temporal_singletask_data(
        train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, test_ids=test_ids
    )

    train_x = train_x.numpy()
    train_y = train_y.numpy()  # (n_train, 145)
    test_x  = test_x.numpy()
    test_y  = test_y.numpy()   # (n_test, 145)

    region_name, idxs, task_names = get_region_y_indices_and_names(mode, roi_to_idx)
    train_y_region = train_y[:, idxs]
    test_y_region = test_y[:, idxs]

    print("NaNs per task:", np.isnan(train_y_region).sum(axis=0))
    print("std per task:", train_y_region.std(axis=0))
    print("mix/max per task:", train_y_region.min(axis=0), train_y_region.max(axis=0))

    y_mean = train_y_region.mean(axis=0, keepdims=True)
    y_std = train_y_region.std(axis=0, keepdims=True)

    num_outputs = len(task_names)

    print("Shape of train_x:", train_x.shape)
    print("Shape of train_y:", train_y_region.shape)

    return train_x, train_y_region, test_x, test_y_region,  corresponding_train_ids, corresponding_test_ids, num_outputs, task_names, region_name


def collate_fn(batch):
    # 'batch' is a list of samples, where each sample is a tuple (input, target, subject_id)
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    subject_ids = [item[2] for item in batch]  # Keep as list or convert to tensor if numeric
    
    return inputs, targets, subject_ids

def _mse_mae_per_task(actuals_list, preds_list):
    mse_per_task = []
    mae_per_task = []
    for k in range(len(actuals_list)):
        mse_per_task.append(mean_squared_error(actuals_list[k], preds_list[k]))
        mae_per_task.append(mean_absolute_error(actuals_list[k], preds_list[k]))
    return mse_per_task, mae_per_task, float(np.mean(mse_per_task)), float(np.mean(mae_per_task))
    
#Plot trajectories
def _safe_filename(s:str) -> str:
    s = str(s)
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)

def _ensure_task_plot_dirs(root_dir:str, num_tasks:int):
    task_dirs=[]
    os.makedirs(root_dir, exist_ok=True)
    for k in range(num_tasks):
        d = os.path.join(root_dir, f"task{k+1}")
        os.makedirs(d, exist_ok=True)
        task_dirs.append(d)
    return task_dirs

def _transpose_task_matrix(x, num_tasks:int):
    if x is None:
        return None
    if x.ndim != 2:
        return x
    if x.shape[0] == num_tasks and x.shape[1] != num_tasks:
        return x.T 
    return x

def save_subject_trajectory_plots(
    subject_id,
    t_np,
    y_true_np,
    y_pred_np,
    y_var_np,
    task_dirs,
    task_names=None,
    dpi=300,
    save_csv=True
    ):

    order = np.argsort(t_np)
    t = t_np[order]
    y_true = y_true_np[order, :]
    y_pred = y_pred_np[order, :]
    y_var = y_var_np[order, :] if y_var_np is not None else None

    subj_safe = _safe_filename(subject_id)
    num_tasks = y_true.shape[1]

    for k in range(num_tasks):
        fig = plt.figure(figsize=(7,4))

        plt.plot(t, y_true[:, k], marker='o', linewidth=2, label="Ground Truth")

        plt.plot(t, y_pred[:, k], marker='x', linestyle="--", linewidth=2, label="Predicted Trajectory")

        #Uncertainty 
        y_std = None
        y_lower = None
        y_upper = None
        if y_var is not None:
            y_std = np.sqrt(np.maximum(y_var[:, k], 0.0))
            y_lower = y_pred[:, k] - 2.0 * y_std
            y_upper = y_pred[:, k] + 2.0 * y_std
            plt.fill_between(t, y_lower, y_upper, alpha=0.2)

        task_label = (task_names[k] if task_names is not None else f"Task {k+1}")
        plt.title(f"Subject {subject_id} | {task_label}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        #Save
        out_png = os.path.join(task_dirs[k], f"{subj_safe}_task{k+1}_csv")
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)

        if save_csv:
            out_csv = os.path.join(task_dirs[k], f"{subj_safe}_task{k+1}.csv")
            if y_std is None:
                df = pd.DataFrame({
                    "time": t,
                    "y_true": y_true[:, k],
                    "y_pred": y_pred[:, k],
                })
            else:
                df = pd.DataFrame({
                    "time": t,
                    "y_true": y_true[:, k],
                    "y_pred": y_pred[:, k],
                    "y_std": y_std,
                    "y_lower_2std": y_lower,
                    "y_upper_2std": y_upper,
                })
            df.to_csv(out_csv, index=False)

# Evaluation
def is_monotonic(sequence, sig):
    seq = np.asarray(sequence)
    if seq.size <= 1:
        return True
    if sig < 0:
        return np.all(np.diff(seq) >= 0)
    else:
        return np.all(np.diff(seq) <= 0)
