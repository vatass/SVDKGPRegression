import re
import pandas as pd
from collections import OrderedDict

# -----------------------------
# REGION DICT (your original)
# -----------------------------
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

# -----------------------------
# Flatten ordering (Mode 0)
# -----------------------------
flattened_regions = []
for _, (region_name, rois) in REGION_ROIS.items():
    for _ in rois:
        flattened_regions.append(region_name)


def parse_file(filepath):

    results = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    current_lambda = None
    current_study = None
    current_mode = None
    current_task_offset = 0

    for line in lines:

        if "Heldout Study:" in line:
            current_study = line.split(":")[1].strip()

        if "Lambda penalty value:" in line:
            current_lambda = float(line.split(":")[1].strip())

        if "Mode:" in line:
            current_mode = int(line.split(":")[1].strip())
            current_task_offset = 0  # reset for new block

        if line.startswith("Task"):

            task_id = int(re.search(r"Task (\d+):", line).group(1))

            # Mode corresponds to region index
            region = REGION_ROIS[current_mode][0]

            mse = float(re.search(r"Test MSE=([-+]?\d*\.?\d+)", line).group(1))
            mae = float(re.search(r"Test Mae=([-+]?\d*\.?\d+)", line).group(1))
            nll = float(re.search(r"NLL=([-+]?\d*\.?\d+)", line).group(1))
            coverage = float(re.search(r"Coverage%=(\d*\.?\d+)", line).group(1))
            width = float(re.search(r"Width=(\d*\.?\d+)", line).group(1))

            results.append({
                "lambda": current_lambda,
                "study": current_study,
                "region": region,
                "mse": mse,
                "mae": mae,
                "nll": nll,
                "coverage": coverage,
                "width": width,
            })

    df = pd.DataFrame(results)

    grouped = (
        df.groupby(["lambda", "study", "region"])
        .mean()
        .reset_index()
    )

    return grouped

def pretty_print_tables(df):

    for lam in sorted(df["lambda"].unique()):
        print("\n" + "="*90)
        print(f"LAMBDA = {lam}")
        print("="*90)

        df_lam = df[df["lambda"] == lam]

        for study in sorted(df_lam["study"].unique()):
            if(study == 'WRAP'):
                continue
            print("\n" + "-"*90)
            print(f"HELDOUT STUDY: {study}")
            print("-"*90)

            table = (
                df_lam[df_lam["study"] == study]
                .set_index("region")[["mse", "mae", "nll", "coverage", "width"]]
                .sort_index()
                .round(4)
            )

            print(table)

df_means = parse_file("results_heldout_total.txt")
pretty_print_tables(df_means)
################################################################################################################

import pandas as pd
import json
import os
from collections import OrderedDict
import numpy as np

# 1. Load the ROI to Index mapping
with open('/home/cbica/Desktop/LongGPClustering/roi_to_idx.json', 'r') as f:
    roi_to_idx = json.load(f)


# 2. Define the Region groupings
REGION_ROIS = OrderedDict({
    "Limbic system": ["MUSE_Volume_100","MUSE_Volume_101","MUSE_Volume_116","MUSE_Volume_117","MUSE_Volume_138", "MUSE_Volume_139","MUSE_Volume_166","MUSE_Volume_167","MUSE_Volume_170","MUSE_Volume_171"],
    "Parietal lobe": ["MUSE_Volume_85","MUSE_Volume_86","MUSE_Volume_106","MUSE_Volume_107","MUSE_Volume_148","MUSE_Volume_149", "MUSE_Volume_168","MUSE_Volume_169","MUSE_Volume_176","MUSE_Volume_177","MUSE_Volume_194","MUSE_Volume_195", "MUSE_Volume_198","MUSE_Volume_199"],
    "Ventricular system": ["MUSE_Volume_4","MUSE_Volume_11", "MUSE_Volume_49","MUSE_Volume_50","MUSE_Volume_51","MUSE_Volume_52"],
    "Temporal lobe": ["MUSE_Volume_87","MUSE_Volume_88","MUSE_Volume_122","MUSE_Volume_123", "MUSE_Volume_132","MUSE_Volume_133","MUSE_Volume_154","MUSE_Volume_155","MUSE_Volume_180","MUSE_Volume_181", "MUSE_Volume_184","MUSE_Volume_185","MUSE_Volume_200","MUSE_Volume_201","MUSE_Volume_202","MUSE_Volume_203", "MUSE_Volume_206", "MUSE_Volume_207"],
    "Occipital lobe": ["MUSE_Volume_83","MUSE_Volume_84","MUSE_Volume_108","MUSE_Volume_109","MUSE_Volume_114","MUSE_Volume_115", "MUSE_Volume_128","MUSE_Volume_129","MUSE_Volume_134","MUSE_Volume_135","MUSE_Volume_144","MUSE_Volume_145", "MUSE_Volume_156","MUSE_Volume_157","MUSE_Volume_160","MUSE_Volume_161","MUSE_Volume_196","MUSE_Volume_197"],
    "Frontal lobe": ["MUSE_Volume_81", "MUSE_Volume_82", "MUSE_Volume_102", "MUSE_Volume_103", "MUSE_Volume_104", "MUSE_Volume_105", "MUSE_Volume_112", "MUSE_Volume_113", "MUSE_Volume_118", "MUSE_Volume_119", "MUSE_Volume_120", "MUSE_Volume_121", "MUSE_Volume_124", "MUSE_Volume_125", "MUSE_Volume_136", "MUSE_Volume_137", "MUSE_Volume_140", "MUSE_Volume_141", "MUSE_Volume_142", "MUSE_Volume_143", "MUSE_Volume_146", "MUSE_Volume_147", "MUSE_Volume_150", "MUSE_Volume_151", "MUSE_Volume_152", "MUSE_Volume_153", "MUSE_Volume_162", "MUSE_Volume_163", "MUSE_Volume_164", "MUSE_Volume_165", "MUSE_Volume_172", "MUSE_Volume_173", "MUSE_Volume_174", "MUSE_Volume_175", "MUSE_Volume_178", "MUSE_Volume_179", "MUSE_Volume_182", "MUSE_Volume_183", "MUSE_Volume_186", "MUSE_Volume_187", "MUSE_Volume_190", "MUSE_Volume_191", "MUSE_Volume_192", "MUSE_Volume_193", "MUSE_Volume_204", "MUSE_Volume_205"]
})

def calculate_mse_nll_from_traj(df):
    """Calculates MSE and NLL from the raw trajectory population file."""
    se = (df['y'] - df['score'])**2
    eps = 1e-12 
    nll = 0.5 * np.log(2 * np.pi * (df['variance'] + eps)) + (se / (2 * (df['variance'] + eps)))
    
    df['tmp_se'] = se
    df['tmp_nll'] = nll
    
    # Average per subject, then return global average for this ROI
    sub_metrics = df.groupby('id').agg({'tmp_se': 'mean', 'tmp_nll': 'mean'})
    return sub_metrics['tmp_se'].mean(), sub_metrics['tmp_nll'].mean()

base_path = "/home/cbica/Desktop/LongGPRegressionBaseline/miccai26/"
heldout_studies = ["ADNI", "BLSA", "AIBL", "CARDIA", "HABS", "OASIS", "PENN", "PreventAD", "WRAP"]

for study in heldout_studies:
    print(f"\n--- Processing Study: {study} ---")
    regional_summary = []
    
    for region_name, roi_list in REGION_ROIS.items():
        metrics_accumulator = []
        found_count = 0
        expected_count = len(roi_list)
        
        for roi_name in roi_list:
            roi_volume = str(roi_name.split("_")[-1])
            if roi_volume in roi_to_idx:
                idx = roi_to_idx[roi_volume]
                
                # 1. Path for MAE, Coverage, Width
                mae_file = f"singletask_MUSE_{idx}_dkgp_mae_per_subject_kfold_looso_{study}.csv"
                # 2. Path for Trajectories (MSE, NLL)
                traj_file = f"singletask_MUSE_{idx}_dkgp_population_looso_{study}.csv"
                
                mae_path = os.path.join(base_path, mae_file)
                traj_path = os.path.join(base_path, traj_file)
                if os.path.exists(mae_path) and os.path.exists(traj_path):
                    # Get MAE, Coverage, Width
                    df_mae = pd.read_csv(mae_path)

                    mae_vals = df_mae[['mae_per_subject','interval','coverage']].mean()
                    
                    # Get MSE and NLL
                    df_traj = pd.read_csv(traj_path)
                    mse_val, nll_val = calculate_mse_nll_from_traj(df_traj)
                    
                    # Combine
                    roi_combined = {
                        'mae': mae_vals['mae_per_subject'],
                        'coverage': mae_vals['coverage'],
                        'interval': mae_vals['interval'],
                        'mse': mse_val,
                        'nll': nll_val
                    }
                    metrics_accumulator.append(roi_combined)
                    found_count += 1
        
        status = "✅ COMPLETE" if found_count == expected_count else f"⚠️ MISSING ({found_count}/{expected_count})"
        print(f"{region_name: <20}: {status} biomarkers included.")

        if metrics_accumulator:
            region_avg = pd.DataFrame(metrics_accumulator).mean()
            regional_summary.append({
                "Region": region_name,
                "MSE": region_avg['mse'],
                "MAE": region_avg['mae'],
                "NLL": region_avg['nll'],
                "Cov.": region_avg['coverage'],
                "Width": region_avg['interval']
            })
    
    if regional_summary:
        print(f"\nAggregated Results for {study}:")
        print(pd.DataFrame(regional_summary).to_string(index=False, float_format="%.4f"))