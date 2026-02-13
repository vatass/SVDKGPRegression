import matplotlib.pyplot as plt
import numpy as np

# Set up professional plot style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2 # Thicker axes frame
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.size'] = 12

# --- SYNTHETIC DATA GENERATION ---
# This simulates the output of your multitask GP model for visualization purposes.
np.random.seed(101) # for reproducibility

# Time grid for smooth predictions
t_grid = np.linspace(0, 10, 200)

# Simulating irregular ground truth observations for a single subject
n_obs = 12
t_obs = np.sort(np.random.uniform(0.5, 9.5, n_obs))

# --- TASK 1: Decreasing Monotonic Biomarker (e.g., Hippocampus Volume) ---
# True underlying trend (monotonic decreasing)
true_trend_1 = 4000 - 1500 * (1 / (1 + np.exp(-(t_grid - 5)))) 
# Ground truth noisy observations
y_obs_1 = np.interp(t_obs, t_grid, true_trend_1) + np.random.normal(0, 100, n_obs)
# GP Prediction (slightly smoother than truth) and growing uncertainty
y_pred_mean_1 = true_trend_1 + np.sin(t_grid/2)*30
y_std_1 = 80 + 15 * t_grid # Uncertainty grows slightly over time

# --- TASK 2: Increasing Monotonic Biomarker (e.g., Ventricular Volume) ---
# True underlying trend (monotonic increasing)
true_trend_2 = 1500 + 2000 * (1 / (1 + np.exp(-(t_grid - 4))))
# Ground truth noisy observations
y_obs_2 = np.interp(t_obs, t_grid, true_trend_2) + np.random.normal(0, 150, n_obs)
# GP Prediction
y_pred_mean_2 = true_trend_2 + np.cos(t_grid/3)*40
y_std_2 = 100 + 10 * t_grid


# --- PLOTTING ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# Colors (using a professional colorblind-friendly palette concept)
color_pred = '#1f77b4'   # Muted blue for prediction
color_band = '#a6cee3'   # Lighter blue for uncertainty
color_obs = '#e31a1c'    # distinct red for observations

# --- Plot Task 1 ---
ax = axes[0]
# 1. The uncertainty band (2 standard deviations)
ax.fill_between(t_grid, 
                y_pred_mean_1 - 2*y_std_1, 
                y_pred_mean_1 + 2*y_std_1,
                color=color_band, alpha=0.4, label='95% GP Confidence Interval', zorder=1)

# 2. The predictive mean
ax.plot(t_grid, y_pred_mean_1, color=color_pred, linewidth=3, label='DKL-GP Predicted Mean', zorder=2)

# 3. The observations
ax.scatter(t_obs, y_obs_1, color=color_obs, s=60, edgecolor='black', linewidth=1, label='Observed Data', zorder=3)

ax.set_title("Task 1: Decreasing Biomarker\n(e.g., Limbic Volume)", fontweight='bold', pad=15)
ax.set_ylabel("Biomarker Volume (mm³)", fontweight='bold')
ax.set_xlabel("Time from Baseline (Years)", fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)

# --- Plot Task 2 ---
ax = axes[1]
# 1. The uncertainty band
ax.fill_between(t_grid, 
                y_pred_mean_2 - 2*y_std_2, 
                y_pred_mean_2 + 2*y_std_2,
                color=color_band, alpha=0.4, zorder=1)

# 2. The predictive mean
ax.plot(t_grid, y_pred_mean_2, color=color_pred, linewidth=3, zorder=2)

# 3. The observations
ax.scatter(t_obs, y_obs_2, color=color_obs, s=60, edgecolor='black', linewidth=1, zorder=3)

ax.set_title("Task 2: Increasing Biomarker\n(e.g., Ventricular System)", fontweight='bold', pad=15)
ax.set_xlabel("Time from Baseline (Years)", fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)

# Add a combined legend to the first plot
axes[0].legend(loc='best', frameon=True, framealpha=0.9, shadow=True)

plt.tight_layout()

# Define the output filename
filename = "multitask_dkgp_visualization.svg"

# SAVE TO SVG
# bbox_inches='tight' ensures no labels are clipped
plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)

print(f"Visualization saved successfully to: {filename}")
plt.close()