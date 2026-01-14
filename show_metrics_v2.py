# ===========================
# Publication-ready plotting for MICCAI
# Single-cell version (drop-in replacement)
# ===========================

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# ---------------------------
# Global publication style
# ---------------------------
def set_pub_style():
    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=1.4
    )
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 1.2,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 12,
        "legend.title_fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "serif",
        "pdf.fonttype": 42,  # editable text
        "ps.fonttype": 42
    })

set_pub_style()

# ---------------------------
# Metric labels (paper-ready)
# ---------------------------
METRIC_LABELS = {
    "mae": "MAE ↓",
    "mse": "MSE ↓",
    "r2": "R² ↑",
    "coverage": "Coverage ↑",
    "interval": "Interval Width ↓",
    "mean_roc_dev": "Mean ROC Deviation ↓"
}

# ---------------------------
# Core plotting helper
# ---------------------------
def plot_metric_bar(metrics_df, metric, output_path):
    order = sorted(metrics_df["penalty"].unique())

    plt.figure(figsize=(7, 4))  # MICCAI 2-column friendly
    ax = sns.barplot(
        data=metrics_df,
        x="penalty",
        y=metric,
        hue="biomarker",
        order=order,
        palette="colorblind",
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_xlabel("Monotonicity Penalty λ")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(METRIC_LABELS.get(metric, metric))

    sns.despine(trim=True)

    ax.legend(
        title="Biomarker",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.05)

    plt.tight_layout()

    for ext in ["pdf", "svg", "png"]:
        plt.savefig(
            os.path.join(output_path, f"{metric}_vs_penalty.{ext}"),
            bbox_inches="tight"
        )

    plt.close()

# ---------------------------
# Main metric plotting
# ---------------------------
def save_bar_plots(base_path: str, output_path: str, biomarkers: list, penalties: list):
    all_metrics = []
    os.makedirs(output_path, exist_ok=True)

    if base_path == 'MUSE':
        for biomarker in biomarkers:
            for penalty in penalties:
                file_path = os.path.join(
                    base_path,
                    f'H_MUSE_Volume_{biomarker}',
                    f'svdk_monotonic_{penalty}_population_fold_metrics.csv'
                )
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df['biomarker'] = str(biomarker)
                    df['penalty'] = float(penalty)
                    all_metrics.append(df.iloc[0])
                else:
                    print(f"[!] File not found: {file_path}")
    else:
        biomarker = biomarkers[0]
        for penalty in penalties:
            file_path = os.path.join(
                str(biomarker),
                f'svdk_monotonic_{penalty}_population_fold_metrics.csv'
            )
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['biomarker'] = str(biomarker)
                df['penalty'] = float(penalty)
                all_metrics.append(df.iloc[0])
            else:
                print(f"[!] File not found: {file_path}")

    if not all_metrics:
        print("[!] No data loaded.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df["penalty"] = metrics_df["penalty"].astype(float)
    metrics_df["biomarker"] = metrics_df["biomarker"].astype(str)

    metrics_to_plot = [
        "mae", "mse", "r2",
        "coverage", "interval",
        "mean_roc_dev"
    ]

    for metric in metrics_to_plot:
        plot_metric_bar(metrics_df, metric, output_path)
        print(f"[✓] Saved {metric} plots to {output_path}")

# ---------------------------
# Monotonicity parsing + plot
# ---------------------------
def parse_monotonicity_txt(base_path: str, biomarkers: list, output_path: str):
    records = []

    for biomarker in biomarkers:
        results_file = os.path.join(
            base_path,
            f'H_MUSE_Volume_{biomarker}',
            'results.txt'
        )

        if not os.path.exists(results_file):
            print(f"[!] Missing: {results_file}")
            continue

        with open(results_file, 'r') as f:
            content = f.read()

        pattern = r'Percentage of samples where monotonicity is achieved:\s*([\d\.]+)% for lambda_penalty\s*([\d\.]+)'
        matches = re.findall(pattern, content)

        for percentage, penalty in matches:
            records.append({
                "biomarker": str(biomarker),
                "penalty": float(penalty),
                "monotonicity": float(percentage)
            })

    if not records:
        print("[!] No monotonicity data found.")
        return

    df = pd.DataFrame(records)
    order = sorted(df["penalty"].unique())

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(
        data=df,
        x="penalty",
        y="monotonicity",
        hue="biomarker",
        order=order,
        palette="colorblind",
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_xlabel("Monotonicity Penalty λ")
    ax.set_ylabel("Monotonicity (%)")
    ax.set_title("Monotonicity Satisfaction")

    sns.despine(trim=True)

    ax.legend(
        title="Biomarker",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        plt.savefig(
            os.path.join(output_path, f"monotonicity_vs_penalty.{ext}"),
            bbox_inches="tight"
        )

    plt.close()
    print(f"[✓] Saved monotonicity plots to {output_path}")

# ===========================
# RUN COMMANDS (unchanged logic)
# ===========================

save_bar_plots(
    base_path='MUSE',
    output_path='./plots/MUSE/penalty_comparison',
    biomarkers=[47, 48, 51, 52],
    penalties=[0.0, 0.75, 1.0, 5.0]
)

save_bar_plots(
    base_path='SPARE_AD',
    output_path='./plots/SPARE_AD',
    biomarkers=['SPARE_AD'],
    penalties=[0.0, 0.75, 1.0, 5.0]
)

save_bar_plots(
    base_path='SPARE_BA',
    output_path='./plots/SPARE_BA',
    biomarkers=['SPARE_BA'],
    penalties=[0.0, 0.75, 1.0, 5.0]
)

save_bar_plots(
    base_path='ADAS',
    output_path='./plots/ADAS',
    biomarkers=['ADAS'],
    penalties=[0.0, 0.75, 1.0, 5.0]
)

save_bar_plots(
    base_path='MMSE',
    output_path='./plots/MMSE',
    biomarkers=['MMSE'],
    penalties=[0.0, 0.75, 1.0, 5.0]
)
