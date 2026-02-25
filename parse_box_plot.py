import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# SETTINGS
# ==========================================================
base_path = "/home/cbica/Desktop/LongGPRegressionBaseline/miccai26/"
results_txt = "results_heldout_total.txt"

heldout_studies = ["ADNI", "BLSA", "AIBL", "CARDIA",
                   "HABS", "OASIS", "PENN", "PreventAD", "WRAP"]

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16
})

# ==========================================================
# PART 1 — SINGLE TASK ROI-LEVEL
# ==========================================================

def calculate_mse_nll_from_traj(df):
    se = (df['y'] - df['score'])**2
    eps = 1e-12
    nll = 0.5 * np.log(2 * np.pi * (df['variance'] + eps)) + (se / (2 * (df['variance'] + eps)))
    df['tmp_se'] = se
    sub_metrics = df.groupby('id').agg({'tmp_se': 'mean'})
    return sub_metrics['tmp_se'].mean()

with open('/home/cbica/Desktop/LongGPClustering/roi_to_idx.json') as f:
    roi_to_idx = json.load(f)

single_results = []

for study in heldout_studies:
    for roi_volume, idx in roi_to_idx.items():

        mae_file = f"singletask_MUSE_{idx}_dkgp_mae_per_subject_kfold_looso_{study}.csv"
        traj_file = f"singletask_MUSE_{idx}_dkgp_population_looso_{study}.csv"

        mae_path = os.path.join(base_path, mae_file)
        traj_path = os.path.join(base_path, traj_file)

        if os.path.exists(mae_path) and os.path.exists(traj_path):

            df_mae = pd.read_csv(mae_path)
            mae_vals = df_mae[['mae_per_subject', 'interval', 'coverage']].mean()

            df_traj = pd.read_csv(traj_path)
            mse_val = calculate_mse_nll_from_traj(df_traj)

            single_results.append({
                "study": study,
                "roi": roi_volume,
                "mse": mse_val,
                "mae": mae_vals['mae_per_subject'],
                "coverage": mae_vals['coverage'],
                "interval": mae_vals['interval']
            })

df_single = pd.DataFrame(single_results)

# ==========================================================
# PART 2 — MULTITASK ROI-LEVEL (TASK = ROI)
# ==========================================================

def parse_multitask_roi(filepath):

    results = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    current_lambda = None
    current_study = None

    for line in lines:

        if "Heldout Study:" in line:
            current_study = line.split(":")[1].strip()

        if "Lambda penalty value:" in line:
            current_lambda = float(line.split(":")[1].strip())

        if line.startswith("Task"):

            task_id = int(re.search(r"Task (\d+):", line).group(1))

            mse = float(re.search(r"Test MSE=([-+]?\d*\.?\d+)", line).group(1))
            mae = float(re.search(r"Test Mae=([-+]?\d*\.?\d+)", line).group(1))
            coverage = float(re.search(r"Coverage%=(\d*\.?\d+)", line).group(1))
            width = float(re.search(r"Width=(\d*\.?\d+)", line).group(1))

            results.append({
                "lambda": current_lambda,
                "study": current_study,
                "roi": task_id,   # task = ROI
                "mse": mse,
                "mae": mae,
                "coverage": coverage,
                "interval": width
            })

    return pd.DataFrame(results)

df_multi_all = parse_multitask_roi(results_txt)

# choose lambda
selected_lambda = df_multi_all["lambda"].unique()[0]
df_multi = df_multi_all[df_multi_all["lambda"] == selected_lambda]

# ==========================================================
# PLOTTING FUNCTION
# ==========================================================
def make_boxplots(df, output_folder, prefix):

    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_folder, exist_ok=True)

    metrics = ["mae", "mse", "coverage", "interval"]
    studies = sorted(df["study"].unique())
    palette = sns.color_palette("Set2", len(studies))

    for metric in metrics:

        plt.figure(figsize=(12, 7))

        # Calculate global min/max for y-axis scaling
        ymin = df[metric].min()
        ymax = df[metric].max()
        padding = (ymax - ymin) * 0.15 if ymax != ymin else 0.1  # avoid zero padding

        sns.boxplot(
            data=df,
            x="study",
            y=metric,
            hue="study",
            palette=palette,
            dodge=False,
            legend=False,
            linewidth=2.8,
            showfliers=False,
            boxprops=dict(edgecolor='black', linewidth=2.5, alpha=0.95),
            medianprops=dict(color="black", linewidth=3.5)
        )

        plt.ylim(ymin - padding, ymax + padding)

        ax = plt.gca()

        ax.set_title(
            f"{prefix} {metric.upper()} per Held-out Study",
            fontsize=20,
            fontweight="bold",
            color="black"
        )

        ax.set_xlabel("Held-out Study",
                      fontsize=16,
                      fontweight="bold",
                      color="black")

        ax.set_ylabel(metric.upper(),
                      fontsize=16,
                      fontweight="bold",
                      color="black")

        ax.tick_params(axis='x', labelsize=13, colors='black')
        ax.tick_params(axis='y', labelsize=13, colors='black')

        # Make tick labels bold
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")

        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        # Stronger axis spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("black")

        sns.despine(offset=5, trim=True)

        plt.xticks(rotation=40)
        plt.tight_layout()

        save_path = os.path.join(
            output_folder,
            f"{prefix.lower()}_{metric}.png"
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

# ==========================================================
# GENERATE ALL PLOTS
# ==========================================================

make_boxplots(df_single, "box_plots_singletask", "SingleTask")
make_boxplots(df_multi, "box_plots_multitask", "MultiTask")