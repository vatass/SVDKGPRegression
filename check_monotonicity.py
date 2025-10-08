import pandas as pd
import numpy as np
import ast
import argparse

parser = argparse.ArgumentParser(description='Evaluate monotonicit aspects and make figures')
parser.add_argument("--data_file", help="Path to save all figures")

args = parser.parse_args()
output_file = args.data_file

def parse_ts(x):
    if isinstance(x,str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return None 
    if not isinstance(x, (list, tuple)):
        return None
    try:
        return [int(v) for v in x]
    except Exception:
        return None

def parse_seq(s):
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    if not s.strip():
        return []
    return [float(x) for x in s.split()]

def parse_num_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None 
    if isinstance(x, (list, tuple)):
        vals = x
    elif isinstance(x, str):
        s = x.strip()
        try:
            vals = ast.literal_eval(s)
            if not isinstance(vals, (list, tuple)):
                raise ValueError
        except Exception:
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1]

            parts = [p for chunk in s.split(',') for p in chunk.split()]
            vals = parts
    else:
        return None
    
    try:
        return [float(v) for v in vals]
    except Exception:
        return None

old = pd.read_csv("{}/svdk_monotonic_0.0_population_per_subject_metrics.csv".format(output_file), usecols=["id", "monotonicity", "timesteps", "sequence","true_sequence","subject_time"])
new = pd.read_csv("{}/svdk_monotonic_0.75_population_per_subject_metrics.csv".format(output_file), usecols=["id", "monotonicity", "timesteps", "sequence", "true_sequence","subject_time"])

for df in (old, new):
    df["monotonicity"] = pd.to_numeric(df["monotonicity"], errors="coerce").fillna(0).astype(int)
    df["timesteps"] = df["timesteps"].apply(parse_ts)
    df["sequence"] = df["sequence"].apply(parse_seq)
    df["true_sequence"] = df["true_sequence"].apply(parse_num_list)
    df["subject_time"] = df["subject_time"].apply(parse_num_list)

m = old.merge(new, on="id", suffixes=("_old", "_new"))

zero_to_one = (m.monotonicity_old == 0) & (m.monotonicity_new == 1) 
one_to_zero = (m.monotonicity_old == 1) & (m.monotonicity_new == 0) 

print("0 to 1 count: {:.2f}%".format(int(zero_to_one.sum())*100/375))
print("1 to 0 count: {:.2f}%".format(int(one_to_zero.sum())*100/375))

ids_0_to_1 = m.loc[zero_to_one, "id"].tolist()
ids_1_to_0 = m.loc[one_to_zero, "id"].tolist()
print("IDS 0 to 1:", ids_0_to_1)
print("IDS 1 to 0:", ids_1_to_0)

#Timestep change analysis
if "timesteps" in new.columns:
    ts_series = new["timesteps"]
elif "timesteps_new" in m.columns:
    ts_series = m["timesteps_new"]
else:
    ts_series = pd.Series([], dtype=object)

norm_changes_all = []
norm_changes_first = []

for ts in ts_series.dropna():
    L = ts[0] if ts else None
    if not L or L<=0:
        continue
    changes = ts[1:]
    if not changes:
        continue

    for p in changes:
        norm = max(0.0, min(1.0, p / L))
        norm_changes_all.append(norm)
    
    first = changes[0]
    norm_changes_first.append(max(0.0, min(1.0, first / L)))

def summarize(label, arr):
    if not arr:
        print("f{label}: no changes found")
        return
    a = np.array(arr)
    early = np.mean((a >= 0.0) & (a < 0.33)) * 100
    middle = np.mean((a >= 0.33) & (a < 0.66)) * 100
    late = np.mean(a >= 0.66) * 100

    hist, edges = np.histogram(a, bins=10, range=(0.0,1.0))
    peak_bin = int(np.argmax(hist))
    peak_span = (edges[peak_bin], edges[peak_bin + 1])
    peak_pct = (hist[peak_bin] / len(a)) * 100

    print(f"--- {label} ---")
    print(f"Early [0.00-0.33]: {early:5.2f}%")
    print(f"Middle [0.33-0.66]: {middle:5.2f}%")
    print(f"Late [0.66]: {late:5.2f}%")
    print(f"Most common bin: [{peak_span[0]:.2f}], {peak_span[1]:.2f}) with {peak_pct:.2f}% of changes\n")

summarize("ALL changes", norm_changes_all)
summarize("First changes", norm_changes_first)

import matplotlib.pyplot as plt
#New plots for plotting with respective timestep
def get_series_by_id(df, ids):
    
    sub = df.loc[df["id"].isin(ids), ["id", "subject_time", "sequence", "true_sequence"]].copy()
    sub = sub.set_index("id")
    data = {}
    for idx, row in sub.iterrows():
        months = row.get("subject_time", None)
        pred = row.get("sequence", None)
        true = row.get("true_sequence", None)
        if isinstance(months, list) and (isinstance(pred, list) or isinstance(true, list)):
            data[idx] = (months, pred, true)
    return data

# First, check if "sequence" column is in old or new or both:
for df_name, df in [("old", old), ("new", new)]:
    if "sequence" not in df.columns:
        print(f'"sequence" column not found in {df_name} DataFrame')

# Let's assume "sequence" is in both old and new DataFrames for plotting.
# You can adjust this if it's only in one.

# Extract sequences for the IDs that went from 0 to 1
# seq_0_to_1_old = old.loc[old["id"].isin(ids_0_to_1), ["id", "sequence"]].set_index("id")["sequence"]
# seq_0_to_1_new = new.loc[new["id"].isin(ids_0_to_1), ["id", "sequence"]].set_index("id")["sequence"]

# # Extract sequences for the IDs that went from 1 to 0
# seq_1_to_0_old = old.loc[old["id"].isin(ids_1_to_0), ["id", "sequence"]].set_index("id")["sequence"]
# seq_1_to_0_new = new.loc[new["id"].isin(ids_1_to_0), ["id", "sequence"]].set_index("id")["sequence"]

# print(seq_0_to_1_old)

# def plot_sequences(old_seq, new_seq, title, name):
#     plt.figure(figsize=(12, 6), dpi=150)
#     for idx in old_seq[0:5].index:
#         # parse sequence if needed
        
#         old_vals = old_seq[idx]
#         new_vals = new_seq.get(idx, None)  # new sequence may be missing for some ids
        
#         if old_vals is not None:
#             plt.plot(old_vals, label=f"{idx} old", linestyle="--")
#         if new_vals is not None:
#             plt.plot(new_vals, label=f"{idx} new")
    
#     plt.title(title)
#     plt.xlabel("Timestep")
#     plt.ylabel("Sequence Value")
#     plt.legend()
#     plt.grid()
#     plt.show()
#     plt.savefig('{}.png'.format(name))
#     plt.savefig('{}.svg'.format(name))

# # Plot sequences for 0 to 1 changes
# if not seq_0_to_1_old.empty and not seq_0_to_1_new.empty:
#     plot_sequences(seq_0_to_1_old, seq_0_to_1_new, "Sequences for IDs Changing from 0 to 1", 'seq_0_to_1')

# # Plot sequences for 1 to 0 changes
# if not seq_1_to_0_old.empty and not seq_1_to_0_new.empty:
#     plot_sequences(seq_1_to_0_old, seq_1_to_0_new, "Sequences for IDs Changing from 1 to 0", 'seq_1_to_0')

def plot_sequences_months(id_to_data, title, name, output_file, max_ids=1, also_plot_old=None):
    plt.figure(figsize=(12, 6), dpi=150)
    shown = 0
    for idx, (months, preds, true) in id_to_data.items():
        if shown >= max_ids:
            break
        months = months or []
        preds = preds or []
        true = true or []

        def trim_xy(x_m, y):
            n = min(len(x_m), len(y))
            return (x_m[:n], y[:n])
        
        if len(months) and len(preds):
            x_p, y_p = trim_xy(months, preds)
            if len(x_p):
                plt.plot(x_p, y_p, label=f"{idx} Pred (new)")

        if len(months) and len(true):
            x_t, y_t = trim_xy(months, true)
            if len(x_t):
                plt.plot(x_t, y_t, linestyle="--", label=f"{idx} True")
        
        if also_plot_old is not None and idx in also_plot_old:
            months_old, pred_old = also_plot_old[idx]
            if isinstance(months_old, list) and isinstance(pred_old, list):
                x_o, y_o = trim_xy(months_old, pred_old)
                if len(x_o):
                    plt.plot(x_o, y_o, linestyle=":", label=f"{idx} Pred (old)")

        shown += 1
    
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Sequence Value")
    plt.legend(loc="upper left", )
    plt.grid()
    plt.show()
    plt.savefig('{}/{}.png'.format(output_file, name))
    plt.savefig('{}/{}.svg'.format(output_file, name))

    return None

new_0to1 = get_series_by_id(new, ids_0_to_1)
new_1to0 = get_series_by_id(new, ids_1_to_0)

def get_old_pred_on_months(df, ids):
    if "subject_time" not in df.columns:
        return {}

    sub = df.loc[df["id"].isin(ids), ["id", "subject_time", "sequence"]].set_index("id")
    out = {}
    for idx, row in sub.iterrows():
        months = row.get("subject_time", None)
        preds = row.get("sequence", None)
        if isinstance(months, list) and isinstance(preds, list):
            out[idx] = (months, preds)
    return out

old_0to1_overlay = get_old_pred_on_months(old, ids_0_to_1)
old_1to0_overlay = get_old_pred_on_months(old, ids_1_to_0)

if len(new_0to1):
    plot_sequences_months(new_0to1,"IDs Changing 0 -> 1: Predicted vs True (months from baseline)", 
    "seq_0_to_1", output_file, also_plot_old=old_0to1_overlay if len(old_0to1_overlay) else None,)

if len(new_1to0):
    plot_sequences_months(new_0to1,"IDs Changing 1 -> 0: Predicted vs True (months from baseline)", 
    "seq_1_to_0", output_file, also_plot_old=old_1to0_overlay if len(old_1to0_overlay) else None,)

