# test.py
import argparse
import datetime
import os
import pickle
import random
from collections import OrderedDict
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# --- Custom Imports ---
from data import CognitiveDataset, TestSubjectBatchSampler, collate_fn
from utils import (
    load_and_preprocess_region_based_data, 
    select_inducing_points, 
    _transpose_task_matrix, 
    save_subject_trajectory_plots, 
    _ensure_task_plot_dirs, 
    is_monotonic
)
from metrics import (
    gaussian_nll_per_task, 
    coverage_and_width_per_task, 
    _mse_mae_per_task
)
from functions import (
    FeatureExtractorLatentConcatenation, 
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

    parser = argparse.ArgumentParser(description='Multi-output GP Regression Evaluation')
    parser.add_argument("--experimentID", help="Indicates the Experiment Identifier", default='adniblsa')
    parser.add_argument("--file", help="Identifier for the data", default="subjectsamples_longclean_mmse_dlmuse_allstudies")
    parser.add_argument("--folder", type=int, default=2)
    parser.add_argument("--sigma", type=float, nargs="+", default=None)
    parser.add_argument("--lambda_val", type=float, default=0.0)
    parser.add_argument("--mode", type=int, default=0, help="Mode for brain region training")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--points", type=int, default=3, help="Points per subject")
    parser.add_argument("--heldout", type=int, default=-1, help="Type of heldout study")
    args = parser.parse_args()

    # Data Loading Logic
    train_ids, test_ids = [], []
    heldout_study = ['ADNI', 'BLSA', 'AIBL', 'CARDIA', 'HABS', 'OASIS', 'PENN', 'PreventAD', 'WRAP'] 

    if args.heldout == -1:
        with (open("/home/cbica/Desktop/DKGP/data/train_subject_allstudies_ids_dl_hmuse" + str(fold) +  ".pkl", "rb")) as openfile:
            while True:
                try: train_ids.append(pickle.load(openfile))
                except EOFError: break 

        with (open("/home/cbica/Desktop/DKGP/data/test_subject_allstudies_ids_dl_hmuse" + str(fold) +  ".pkl", "rb")) as openfile:
            while True:
                try: test_ids.append(pickle.load(openfile))
                except EOFError: break
    else:
        with (open("/home/cbica/Desktop/LongGPClustering/data1/train_subject_allstudies_ids_hmuse_" + heldout_study[args.heldout] +  ".pkl", "rb")) as openfile:
            while True:
                try: train_ids.append(pickle.load(openfile))
                except EOFError: break 

        with (open("/home/cbica/Desktop/LongGPClustering/data1/test_subject_allstudies_ids_hmuse_" + heldout_study[args.heldout] +  ".pkl", "rb")) as openfile:
            while True:
                try: test_ids.append(pickle.load(openfile))
                except EOFError: break

    train_ids, test_ids = train_ids[0], test_ids[0]

    # Preprocessing
    train_x, _, test_x, test_y, corresponding_train_ids, corresponding_test_ids, \
    num_outputs, task_names, region_name = load_and_preprocess_region_based_data(
        folder=args.folder, file=args.file, train_ids=train_ids, test_ids=test_ids, mode=args.mode
    )
    
    temporal_index = -1
    test_x = np.hstack((test_x[:, :-1], test_x[:, temporal_index].reshape(-1, 1)))
    test_x = torch.tensor(test_x, dtype=torch.float64)
    test_y = torch.tensor(test_y, dtype=torch.float64)

    train_x = np.hstack((train_x[:, :-1], train_x[:, temporal_index].reshape(-1, 1)))
    train_x = torch.tensor(train_x, dtype=torch.float64)

    # Dataset & DataLoader
    test_dataset = CognitiveDataset(inputs=test_x, targets=test_y, subject_ids=corresponding_test_ids)
    test_subject_sampler = TestSubjectBatchSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_sampler=test_subject_sampler, collate_fn=collate_fn, pin_memory=False, num_workers=0)

    # Load Feature Extractor
    input_dim = test_x.shape[1]
    output_file = "./multitask_trials"
    feature_extractor_gp = FeatureExtractorLatentConcatenation(input_dim, args.hidden_dim).to(device)
    feature_extractor_gp.load_state_dict(torch.load(f'{output_file}/multitask_feature_extractor_latentconcatenation.pth', weights_only=True))
    feature_extractor_gp = feature_extractor_gp.double().eval()
    for p in feature_extractor_gp.parameters(): p.requires_grad = False

    # Initialize GP
    unique_train_subject_ids = list(set(corresponding_train_ids))
    selected_subject_ids = random.sample(unique_train_subject_ids, 128) 
    inducing_points = select_inducing_points(train_x, corresponding_train_ids, selected_subject_ids=selected_subject_ids, num_points_per_subject=args.points, device='cuda')
    inducing_points = inducing_points.double().to(device)

    gp_regression_model = MultitaskDeepKernelGPModel(inducing_points, num_tasks=num_outputs, feature_extractor=feature_extractor_gp).double().to(device)
    regression_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_outputs).double().to(device)

    # Load GP state dictates
    save_file = "./multitask_trials/ckpts/"
    gp_regression_model.load_state_dict(torch.load(f"{save_file}gp_regression_model_{args.mode}_{args.lambda_val}.pth", weights_only=True))
    regression_likelihood.load_state_dict(torch.load(f"{save_file}regression_likelihood_{args.mode}_{args.lambda_val}.pth", weights_only=True))

    model_wrapper = GPModelWrapper(gp_regression_model, regression_likelihood).to(device)
    
    # Setup eval logic
    model_wrapper.eval()
    regression_likelihood.eval()

    plots_root = os.path.join(output_file, f"test_trajectories_mode{args.mode}_{args.lambda_val}_{region_name.replace(' ', '_')}")
    task_plots_dirs = _ensure_task_plot_dirs(plots_root, num_outputs)

    sigma = torch.tensor([-1] * num_outputs if args.mode == 2 else [1] * num_outputs, dtype=torch.float64, device=device)

    print("Phase 1: Running Inference & Saving All Trajectories...")
    all_trajectories_data = []

    with torch.no_grad():
        for inputs, targets, subject_ids in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            subject_id = subject_ids[0]

            gp_regression_output = model_wrapper(inputs)    
            pred_regression = regression_likelihood(gp_regression_output)
            mean_pred = pred_regression.mean 
            var_pred = pred_regression.variance

            mean_np = _transpose_task_matrix(mean_pred.detach().cpu().numpy(), num_outputs)
            var_np = _transpose_task_matrix(var_pred.detach().cpu().numpy(), num_outputs)
            t_np = inputs[:, -1].detach().cpu().numpy()
            y_true_np = targets.detach().cpu().numpy()

            # 1. Save PNG plots for EVERY subject (removed the MAX limit)
            save_subject_trajectory_plots(
                subject_id=subject_id, 
                t_np=t_np, 
                y_true_np=y_true_np, 
                y_pred_np=mean_np, 
                y_var_np=var_np, 
                task_dirs=task_plots_dirs, 
                task_names=task_names
            )

            # 2. Store predictions to construct a master DataFrame
            for i in range(len(t_np)):
                row_dict = {
                    "Subject_ID": subject_id,
                    "Time": t_np[i]
                }
                for k in range(num_outputs):
                    row_dict[f"Task_{k}_True"] = y_true_np[i, k]
                    row_dict[f"Task_{k}_Pred"] = mean_np[i, k]
                    row_dict[f"Task_{k}_Var"]  = var_np[i, k]
                all_trajectories_data.append(row_dict)

    # Save the consolidated raw trajectories
    traj_df = pd.DataFrame(all_trajectories_data)
    traj_csv_path = os.path.join(plots_root, "all_test_trajectories.csv")
    traj_df.to_csv(traj_csv_path, index=False)
    print(f"Saved consolidated test trajectories to: {traj_csv_path}")


    # =====================================================================
    # Phase 2: Calculate Metrics from the Generated Trajectories
    # =====================================================================
    print("Phase 2: Computing metrics from the generated trajectories...")
    df_loaded = pd.read_csv(traj_csv_path)

    # Reconstruct Y Matrices
    Y_true = df_loaded[[f"Task_{k}_True" for k in range(num_outputs)]].values
    Y_mean = df_loaded[[f"Task_{k}_Pred" for k in range(num_outputs)]].values
    Y_var  = df_loaded[[f"Task_{k}_Var" for k in range(num_outputs)]].values

    # 1. Base Metrics
    nll_per_task, nll_mean = gaussian_nll_per_task(Y_true, Y_mean, Y_var)
    cov_per_task, width_per_task, cov_mean, width_mean = coverage_and_width_per_task(Y_true, Y_mean, Y_var, z=1.96)
    
    regression_actuals = [Y_true[:, k] for k in range(num_outputs)]
    regression_predictions = [Y_mean[:, k] for k in range(num_outputs)]
    mse_per_task, mae_per_task, mse_mean, mae_mean = _mse_mae_per_task(regression_actuals, regression_predictions)

    # 2. Monotonicity Checks (grouped by subject)
    mono_subject_ok = [0] * num_outputs
    mono_sample_ok = [0] * num_outputs
    mono_subject_total = 0
    mono_sample_total = 0

    for subject_id, group_df in df_loaded.groupby("Subject_ID"):
        group_df = group_df.sort_values(by="Time")
        mono_subject_total += 1
        
        # Determine number of sample transitions for this subject
        num_transitions = max(0, len(group_df) - 1)
        mono_sample_total += num_transitions

        for k in range(num_outputs):
            seq_k = group_df[f"Task_{k}_Pred"].values
            
            # Subject-level Monotonicity Check
            if is_monotonic(seq_k, sigma[k].item()):
                mono_subject_ok[k] += 1
            
            # Sample-level (df/dt) Check
            if num_transitions > 0:
                delta = seq_k[1:] - seq_k[:-1]
                if sigma[k].item() < 0:
                    mono_sample_ok[k] += np.sum(delta >= 0)
                else:
                    mono_sample_ok[k] += np.sum(delta <= 0)

    # =====================================================================
    # Write final results to CSV files
    # =====================================================================
    
    # Clean up parameters for file naming
    heldout_name = heldout_study[args.heldout]
    sigma_val = int(sigma[0].item()) # assuming all tasks share the same sigma sign based on your code
    lambda_val = args.lambda_val
    
    # 1. Overall Metrics CSV
    overall_metrics_dict = {
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Mode": [args.mode],
        "Mean_Test_MSE": [mse_mean],
        "Mean_Test_MAE": [mae_mean],
        "Mean_Test_NLL": [nll_mean],
        "Mean_Coverage_Pct": [cov_mean],
        "Mean_Width": [width_mean]
    }
    
    df_overall = pd.DataFrame(overall_metrics_dict)
    
    # File naming format: overall_metrics_Heldout_Numtasks_Sigma_Lambda.csv
    overall_csv_name = f"overall_metrics_{heldout_name}_{num_outputs}_{sigma_val}_{lambda_val}.csv"
    overall_csv_path = os.path.join(output_file, overall_csv_name)
    
    # If it exists, append without writing headers again. If not, create it.
    if os.path.exists(overall_csv_path):
        df_overall.to_csv(overall_csv_path, mode='a', header=False, index=False)
    else:
        df_overall.to_csv(overall_csv_path, index=False)
        
    print(f"Overall metrics saved to: {overall_csv_path}")

    # 2. Per-Task Metrics CSV
    per_task_data = []
    for k in range(num_outputs):
        subj_pct = 100.0 * mono_subject_ok[k] / max(mono_subject_total, 1)
        samp_pct = 100.0 * mono_sample_ok[k] / max(mono_sample_total, 1)
        
        per_task_data.append({
            "Task_ID": k + 1,
            "Task_Name": task_names[k] if task_names is not None else f"Task_{k+1}",
            "Test_MSE": mse_per_task[k],
            "Test_MAE": mae_per_task[k],
            "NLL": nll_per_task[k],
            "Coverage_Pct": cov_per_task[k],
            "Width": width_per_task[k],
            "Monotonic_Subjects_Pct": subj_pct,
            "Monotonic_Samples_Pct": samp_pct
        })
        
    df_per_task = pd.DataFrame(per_task_data)
    
    # File naming format: per_task_metrics_Heldout_Numtasks_Sigma_Lambda.csv
    per_task_csv_name = f"per_task_metrics_{heldout_name}_{num_outputs}_{sigma_val}_{lambda_val}.csv"
    per_task_csv_path = os.path.join(output_file, per_task_csv_name)
    
    # Save per-task metrics (overwriting if it exists for this specific run configuration)
    df_per_task.to_csv(per_task_csv_path, index=False)
    print(f"Per-task metrics saved to: {per_task_csv_path}")

    print(f"\nMean Test MSE (avg across tasks): {mse_mean:.4f}")
    print(f"Mean Test MAE (avg across tasks): {mae_mean:.4f}")

if __name__ == "__main__":
    main()