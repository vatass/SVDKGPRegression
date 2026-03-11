# train.py
import argparse
import datetime
import json
import math
import os
import pickle
import random
from collections import OrderedDict

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Sampler

from data import CognitiveDataset, SubjectBatchSampler, collate_fn

from utils import (
    load_and_preprocess_region_based_data, 
    select_inducing_points
)

from functions import (
    FeatureExtractorLatentConcatenation, 
    RegressionNNLatentConcatenation, 
    MultitaskDeepKernelGPModel, 
    GPModelWrapper
)


torch.set_default_dtype(torch.float64)

REGION_ROIS = OrderedDict({
    0: ("Limbic system", [
        "MUSE_Volume_100","MUSE_Volume_101","MUSE_Volume_116","MUSE_Volume_117","MUSE_Volume_138",
        "MUSE_Volume_139","MUSE_Volume_166","MUSE_Volume_167","MUSE_Volume_170","MUSE_Volume_171",
    ]),
    1: ("Parietal lobe", [
        "MUSE_Volume_85","MUSE_Volume_86","MUSE_Volume_106","MUSE_Volume_107","MUSE_Volume_148","MUSE_Volume_149",
        "MUSE_Volume_168","MUSE_Volume_169","MUSE_Volume_176","MUSE_Volume_177","MUSE_Volume_194","MUSE_Volume_195",
        "MUSE_Volume_198","MUSE_Volume_199",
    ]),
    2: ("Ventricular system", [
        "MUSE_Volume_4","MUSE_Volume_11", "MUSE_Volume_49","MUSE_Volume_50","MUSE_Volume_51","MUSE_Volume_52",
    ]),
    3: ("Temporal lobe", [
        "MUSE_Volume_87","MUSE_Volume_88","MUSE_Volume_122","MUSE_Volume_123",
        "MUSE_Volume_132","MUSE_Volume_133","MUSE_Volume_154","MUSE_Volume_155","MUSE_Volume_180","MUSE_Volume_181",
        "MUSE_Volume_184","MUSE_Volume_185","MUSE_Volume_200","MUSE_Volume_201","MUSE_Volume_202","MUSE_Volume_203",
        "MUSE_Volume_206", "MUSE_Volume_207"
    ]),
    4: ("Occipital lobe", [
        "MUSE_Volume_83","MUSE_Volume_84","MUSE_Volume_108","MUSE_Volume_109","MUSE_Volume_114","MUSE_Volume_115",
        "MUSE_Volume_128","MUSE_Volume_129","MUSE_Volume_134","MUSE_Volume_135","MUSE_Volume_144","MUSE_Volume_145",
        "MUSE_Volume_156","MUSE_Volume_157","MUSE_Volume_160","MUSE_Volume_161","MUSE_Volume_196","MUSE_Volume_197",
    ]),
    5: ("Frontal lobe", [
        "MUSE_Volume_81", "MUSE_Volume_82", "MUSE_Volume_102", "MUSE_Volume_103", "MUSE_Volume_104", "MUSE_Volume_105",
        "MUSE_Volume_112", "MUSE_Volume_113", "MUSE_Volume_118", "MUSE_Volume_119", "MUSE_Volume_120", "MUSE_Volume_121",
        "MUSE_Volume_124", "MUSE_Volume_125", "MUSE_Volume_136", "MUSE_Volume_137", "MUSE_Volume_140", "MUSE_Volume_141", 
        "MUSE_Volume_142", "MUSE_Volume_143", "MUSE_Volume_146", "MUSE_Volume_147", "MUSE_Volume_150", "MUSE_Volume_151",
        "MUSE_Volume_152", "MUSE_Volume_153", "MUSE_Volume_162", "MUSE_Volume_163", "MUSE_Volume_164", "MUSE_Volume_165",
        "MUSE_Volume_172", "MUSE_Volume_173", "MUSE_Volume_174", "MUSE_Volume_175", "MUSE_Volume_178", "MUSE_Volume_179",
        "MUSE_Volume_182", "MUSE_Volume_183", "MUSE_Volume_186", "MUSE_Volume_187", "MUSE_Volume_190", "MUSE_Volume_191",
        "MUSE_Volume_192", "MUSE_Volume_193", "MUSE_Volume_204", "MUSE_Volume_205"
    ])
})

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fold = 0

    parser = argparse.ArgumentParser(description='Multi-output GP Regression Training')
    parser.add_argument("--experimentID", help="Indicates the Experiment Identifier", default='adniblsa')
    parser.add_argument("--file", help="Identifier for the data", default="subjectsamples_longclean_mmse_dlmuse_allstudies")
    parser.add_argument("--folder", type=int, default=2)
    parser.add_argument("--sigma", type=float, nargs="+", default=None)
    parser.add_argument("--lambda_val", type=float, default=0.0)
    parser.add_argument("--mode", type=int, default=0, help="Mode for brain region training")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--points", type=int, default=3, help="Points per subject")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for pre-training")
    parser.add_argument("--heldout", type=int, default=-1, help="Type of heldout study")
    args = parser.parse_args()

    # Data Loading Logic
    train_ids, test_ids = [], []
    heldout_study = ['ADNI', 'BLSA', 'AIBL', 'CARDIA', 'HABS', 'OASIS', 'PENN', 'PreventAD', 'WRAP'] 

    if args.heldout == -1:
        print("Loading all studies...")
        with (open("/home/cbica/Desktop/DKGP/data/train_subject_allstudies_ids_dl_hmuse" + str(fold) +  ".pkl", "rb")) as openfile:
            while True:
                try: train_ids.append(pickle.load(openfile))
                except EOFError: break 

        with (open("/home/cbica/Desktop/DKGP/data/test_subject_allstudies_ids_dl_hmuse" + str(fold) +  ".pkl", "rb")) as openfile:
            while True:
                try: test_ids.append(pickle.load(openfile))
                except EOFError: break
    else:
        print("Loading heldout study {}...".format(heldout_study[args.heldout]))
        with (open("/home/cbica/Desktop/LongGPClustering/data1/train_subject_allstudies_ids_hmuse_" + heldout_study[args.heldout] +  ".pkl", "rb")) as openfile:
            while True:
                try: train_ids.append(pickle.load(openfile))
                except EOFError: break 

        with (open("/home/cbica/Desktop/LongGPClustering/data1/test_subject_allstudies_ids_hmuse_" + heldout_study[args.heldout] +  ".pkl", "rb")) as openfile:
            while True:
                try: test_ids.append(pickle.load(openfile))
                except EOFError: break

    train_ids = train_ids[0]
    test_ids = test_ids[0]
    print('Train IDs:', len(train_ids))
    print('Test IDs:', len(test_ids))

    # Preprocessing
    train_x, train_y, test_x, test_y, corresponding_train_ids, corresponding_test_ids, \
    num_outputs, task_names, region_name = load_and_preprocess_region_based_data(
        folder=args.folder, file=args.file, train_ids=train_ids, test_ids=test_ids, mode=args.mode
    )
    
    temporal_index = -1
    train_x = np.hstack((train_x[:, :-1], train_x[:, temporal_index].reshape(-1, 1)))
    train_x = torch.tensor(train_x, dtype=torch.float64)
    train_y = torch.tensor(train_y, dtype=torch.float64)

    print("Train x shape :", train_x.shape)
    print("Train y shape :", train_y.shape)

    # Monotonicity Hyper-parameters
    num_tasks = num_outputs
    sigma = torch.tensor([-1] * num_outputs if args.mode == 2 else [1] * num_outputs, dtype=torch.float64, device=device)
    lambda_penalty = torch.tensor([args.lambda_val] * num_outputs, dtype=torch.float64, device=device)

    # Dataset & DataLoader
    train_dataset = CognitiveDataset(inputs=train_x, targets=train_y, subject_ids=corresponding_train_ids)
    train_sampler = SubjectBatchSampler(train_dataset, batch_size=16, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, pin_memory=False, num_workers=0)

    # Setup Pre-training Model
    input_dim = train_x.shape[1]
    feature_extractor = FeatureExtractorLatentConcatenation(input_dim, args.hidden_dim)
    model = RegressionNNLatentConcatenation(feature_extractor, output_dim=num_outputs).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # 1. Pre-train feature extractor
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch+1}/{args.epochs}, Regression Loss: {running_loss / len(train_dataset):.4f}")

    # Save feature extractor
    output_file = "./multitask_trials"
    os.makedirs(output_file, exist_ok=True)
    torch.save(feature_extractor.state_dict(), f'{output_file}/multitask_feature_extractor_latentconcatenation.pth')

    # Load into frozen GP feature extractor
    feature_extractor_gp = FeatureExtractorLatentConcatenation(input_dim, args.hidden_dim).to(device)
    feature_extractor_gp.load_state_dict(torch.load(f'{output_file}/multitask_feature_extractor_latentconcatenation.pth'))
    feature_extractor_gp = feature_extractor_gp.double().eval()
    for p in feature_extractor_gp.parameters(): p.requires_grad = False

    # Inducing Points
    unique_train_subject_ids = list(set(corresponding_train_ids))
    selected_subject_ids = random.sample(unique_train_subject_ids, 128)
    inducing_points = select_inducing_points(train_x, corresponding_train_ids, selected_subject_ids=selected_subject_ids, num_points_per_subject=args.points, device='cuda')
    inducing_points = inducing_points.double().to(device)

    # Initialize GP
    gp_regression_model = MultitaskDeepKernelGPModel(inducing_points, num_tasks=num_outputs, feature_extractor=feature_extractor_gp).double().to(device)
    regression_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_outputs).double().to(device)
    model_wrapper = GPModelWrapper(gp_regression_model, regression_likelihood).to(device)
    
    mll_regression = gpytorch.mlls.VariationalELBO(regression_likelihood, gp_regression_model, num_data=len(train_dataset), combine_terms=True)

    optimizer_gp = torch.optim.Adam([
        {'params': regression_likelihood.parameters()},
        {'params': gp_regression_model.mean_module.parameters()},
        {'params': gp_regression_model.covar_module.parameters()},
        {'params': gp_regression_model.variational_strategy.parameters()},
    ], lr=1e-3)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gp, step_size=50, gamma=0.5)

    # 2. Train GP
    num_gp_epochs = 150
    for epoch in range(num_gp_epochs):
        model_wrapper.train()
        regression_likelihood.train()
        running_loss = 0.0

        for inputs, targets, subject_ids in train_loader:
            if inputs.shape[0] < 2: continue
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_gp.zero_grad()
            
            f = model_wrapper(inputs)
            pred = regression_likelihood(f)
            loss_regression = -mll_regression(f, targets)

            total_penalty = 0.0
            if args.lambda_val > 0:
                mean_pred = pred.mean
                delta = mean_pred[1:] - mean_pred[:-1]
                same_subject = torch.tensor([subject_ids[i] == subject_ids[i-1] for i in range(1, len(subject_ids))], device=device)
                delta = delta[same_subject]
                pen = torch.relu(sigma.view(1, -1) * delta).mean(dim=0)
                total_penalty = (lambda_penalty * pen).sum()

            total_loss = loss_regression + total_penalty
            total_loss.backward()
            optimizer_gp.step()
            running_loss += total_loss.item() 

        scheduler.step()
        print(f"GP Epoch {epoch+1}/{num_gp_epochs}, Total Loss: {running_loss / len(train_loader):.4f} ")

    # Save GP Models
    save_file = "./multitask_trials/ckpts/"
    os.makedirs(save_file, exist_ok=True)
    torch.save(gp_regression_model.state_dict(), f"{save_file}gp_regression_model_{args.mode}_{args.lambda_val}.pth")
    torch.save(regression_likelihood.state_dict(), f"{save_file}regression_likelihood_{args.mode}_{args.lambda_val}.pth")
    print(f"Models saved to {save_file}")

if __name__ == "__main__":
    main()