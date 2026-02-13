import os
import re
import glob
import pandas as pd
import numpy as np
from collections import OrderedDict


# ==========================
# CONFIG
# ==========================

CSV_DIR = "/home/cbica/Desktop/LongGPRegressionBaseline/miccai26"
CSV_PATTERN = "singletask_unharmonized_dlmuse_MUSE_*_dkgp_mae_kfold_allstudies.csv"
TXT_PATH = "/home/cbica/Desktop/SVDKRegression/multitask_trials/results.txt"  # <-- CHANGE THIS


REGION_ROIS = OrderedDict({
    0: ("Limbic system", ["100","101","116","117","138","139","166","167","170","171"]),
    1: ("Parietal lobe", ["85","86","106","107","148","149","168","169","176","177",
                           "194","195","198","199"]),
    2: ("Ventricular system", ["4","11","49","50","51","52"]),
    3: ("Temporal lobe", ["87","88","122","123","132","133","154","155",
                           "180","181","184","185","200","201","202","203","206","207"]),
    4: ("Occipital lobe", ["83","84","108","109","114","115","128","129",
                           "134","135","144","145","156","157","160","161","196","197"]),
    5: ("Frontal lobe", ["81","82","102","103","104","105","112","113",
                         "118","119","120","121","124","125","136","137",
                         "140","141","142","143","146","147","150","151",
                         "152","153","162","163","164","165","172","173",
                         "174","175","178","179","182","183","186","187",
                         "190","191","192","193","204","205"])
})

# =========================================================
# SINGLE-TASK (FOLD 0 ONLY)
# =========================================================

def compute_single_task_region_fold0():
    csv_files = glob.glob(os.path.join(CSV_DIR, CSV_PATTERN))
    if len(csv_files) == 0:
        raise RuntimeError("No CSV files found. Check path/pattern.")

    roi_metrics = {}
    for file in csv_files:
        match = re.search(r"MUSE_(\d+)_", file)
        if not match:
            continue
        roi_id = match.group(1)
        df = pd.read_csv(file)
        df_fold0 = df[df["kfold"] == 0]
        if df_fold0.empty:
            continue
        row = df_fold0.iloc[0]
        roi_metrics[roi_id] = {
            "MAE": float(row["mae"]),
            "MSE": float(row["mse"]),
            "Coverage": float(row["coverage"] * 100),
            "Width": float(row["interval"])
        }

    region_results = {}
    for mode, (region_name, roi_list) in REGION_ROIS.items():
        available = [roi_metrics[r] for r in roi_list if r in roi_metrics]
        if not available:
            continue
        region_results[mode] = {
            "region": region_name,
            "MAE": np.mean([x["MAE"] for x in available]),
            "MSE": np.mean([x["MSE"] for x in available]),
            "Coverage": np.mean([x["Coverage"] for x in available]),
            "Width": np.mean([x["Width"] for x in available])
        }

    return region_results

# =========================================================
# MULTITASK PARSER (ROBUST)
# =========================================================

def parse_multitask_results_per_region(txt_path):
    if not os.path.exists(txt_path):
        raise RuntimeError("TXT file not found.")

    results = {}
    current_mode = None
    reading_multitask_block = False
    mt_block = {}

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Detect region line
            region_match = re.match(r"Region:\s*(.+)\s+\(Mode (\d+)\)", line)
            if region_match:
                region_name = region_match.group(1).strip()
                current_mode = int(region_match.group(2))
                mt_block = {}
                reading_multitask_block = False
                continue

            # Detect Multitask block (case insensitive, allow spaces)
            if re.match(r"Multitask\s*:", line, re.IGNORECASE):
                reading_multitask_block = True
                continue

            # If we are inside a Multitask block, read metrics
            if reading_multitask_block and current_mode is not None:
                key_val_match = re.match(r"(MAE|MSE|Coverage|Width)\s*:\s*([\d\.\-]+)", line)
                if key_val_match:
                    key = key_val_match.group(1)
                    value = float(key_val_match.group(2))
                    mt_block[key] = value
                # Once all four metrics are read, save
                if len(mt_block) == 4:
                    results[current_mode] = mt_block.copy()
                    reading_multitask_block = False
                    mt_block = {}

    return results

# =========================================================
# MAIN COMPARISON
# =========================================================

def main():
    single_task = compute_single_task_region_fold0()
    multitask = parse_multitask_results_per_region(TXT_PATH)

    print("\n=========== REGION COMPARISON (FOLD 0 ONLY) ===========\n")

    for mode in sorted(single_task.keys()):
        if mode not in multitask:
            continue

        region_name = single_task[mode]["region"]

        st = single_task[mode]
        mt = multitask[mode]

        print(f"\nRegion: {region_name} (Mode {mode})")
        print("-" * 50)

        print("Single-task (Fold 0):")
        print(f"  MAE      : {st['MAE']:.6f}")
        print(f"  MSE      : {st['MSE']:.6f}")
        print(f"  Coverage : {st['Coverage']:.6f}")
        print(f"  Width    : {st['Width']:.6f}")

        print("\nMultitask:")
        print(f"  MAE      : {mt['MAE']:.6f}")
        print(f"  MSE      : {mt['MSE']:.6f}")
        print(f"  Coverage : {mt['Coverage']:.6f}")
        print(f"  Width    : {mt['Width']:.6f}")

        print("\nDifference (Multi - Single):")
        print(f"  MAE      : {(mt['MAE'] - st['MAE']):.6f}")
        print(f"  MSE      : {(mt['MSE'] - st['MSE']):.6f}")
        print(f"  Coverage : {(mt['Coverage'] - st['Coverage']):.6f}")
        print(f"  Width    : {(mt['Width'] - st['Width']):.6f}")

    print("\nComparison complete.\n")

if __name__ == "__main__":
    main()
