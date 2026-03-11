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


# Step 2: Define the FeatureExtractor class

class FeatureExtractorLatentConcatenation(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractorLatentConcatenation, self).__init__()
        # Assuming the last feature is time
        self.imaging_dim = input_dim - 1
        self.time_dim = 1
        self.hidden_dim = hidden_dim

        # Layers for imaging features
        self.imaging_fc = nn.Sequential(
            nn.Linear(self.imaging_dim, hidden_dim),
            nn.ReLU(),
            # You can add more layers if needed
        )

        # Layers for time feature
        self.time_fc = nn.Sequential(
            nn.Linear(self.time_dim, hidden_dim),
            nn.ReLU(),
            # You can add more layers if needed
        )

        # Final layers after concatenation
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            # You can add more layers if needed
        )

    def forward(self, x):
        x_imaging = x[:, :-1]  # All features except the last one
        x_time = x[:, -1].unsqueeze(1)  # The last feature (time)

        # Process imaging features
        imaging_features = self.imaging_fc(x_imaging)

        # Process time feature
        time_features = self.time_fc(x_time)

        # Concatenate imaging and time features
        combined_features = torch.cat([imaging_features, time_features], dim=1)

        # Final processing
        output = self.fc_combined(combined_features)
        return output

class RegressionNNLatentConcatenation(nn.Module):
    def __init__(self, feature_extractor, output_dim=4):
        super(RegressionNNLatentConcatenation, self).__init__()
        self.feature_extractor = feature_extractor
    
        # The output dimension of feature_extractor is hidden_dim
        self.fc_out = nn.Linear(feature_extractor.hidden_dim, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.fc_out(features)
        return output


# Step 3: Define the Multitask GP Regression Model
class MultitaskDeepKernelGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks, feature_extractor):
        batch_shape = torch.Size([num_tasks])  # For multitask GPs
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=batch_shape
        )
        base_variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            base_variational_strategy, num_tasks=num_tasks
        )
        super(MultitaskDeepKernelGPModel, self).__init__(variational_strategy)

        self.feature_extractor = feature_extractor  # Shared feature extractor
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
            batch_shape=batch_shape
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Step 5: Define the GPModelWrapper
class GPModelWrapper(nn.Module):
    def __init__(self, gp_model, likelihood):
        super(GPModelWrapper, self).__init__()
        self.gp_model = gp_model
        self.likelihood = likelihood

    def forward(self, x):
        return self.gp_model(x)