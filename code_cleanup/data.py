# svdk_regression_multitask.py
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

def select_region_targets(Y: np.ndarray, mode: int, roi_to_idx: dict):

    if Y.ndim != 2 or Y.shape[1] != 145:
        raise ValueError(f"Expected Y shape (n,145). Got {Y.shape}")
    idxs = get_region_y_indices(mode, roi_to_idx)
    return Y[:, idxs]


# Step 1: Define the CognitiveDataset class
class CognitiveDataset(Dataset):
    def __init__(self, inputs, targets, subject_ids):
        """
        inputs: NumPy array of input features, shape (num_samples, input_dim)
        targets: NumPy array of target values, shape (num_samples,)
        subject_ids: List or array of subject IDs, length num_samples
        """
        assert len(inputs) == len(targets) == len(subject_ids), "Inputs, targets, and subject_ids must have the same length."

        self.inputs = torch.tensor(inputs, dtype=torch.float64)
        self.targets = torch.tensor(targets, dtype=torch.float64)
        self.subject_ids = subject_ids  # List or array of subject IDs

        # Create a mapping from subject ID to indices
        self.subject_to_indices = {}
        for idx, subject_id in enumerate(self.subject_ids):
            if subject_id not in self.subject_to_indices:
                self.subject_to_indices[subject_id] = []
            self.subject_to_indices[subject_id].append(idx)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.subject_ids[idx]
    
class SubjectBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subject_ids = list(dataset.subject_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.subject_ids)
        batch = []
        for subject_id in self.subject_ids:
            indices = self.dataset.subject_to_indices[subject_id]
            batch.extend(indices)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class TestSubjectBatchSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.subject_ids = list(dataset.subject_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.subject_ids)
        for subject_id in self.subject_ids:
            indices = self.dataset.subject_to_indices[subject_id]
            yield indices  # Yield all indices for the current subject