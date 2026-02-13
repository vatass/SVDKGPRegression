import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_DIR = "multitask_trials"
OUTPUT_FILE = "region_grid.png"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def get_region_folders(base_dir):
    return [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("test_trajectories_mode") and
        os.path.isdir(os.path.join(base_dir, d))
    ]


def extract_region_name(folder_name):
    # Region is everything after the last numeric mode value
    parts = folder_name.split("_")
    # Region is last two parts (e.g. Occipital_lobe)
    return "_".join(parts[-2:])


def extract_biomarker_id(filename):
    # Example: 002_S_1155_task1_csv.png -> 1155
    return filename.split("_")[2]


# ---- Main logic ----

region_folders = get_region_folders(BASE_DIR)
region_folders.sort()  # consistent order

n_regions = len(region_folders)
cols = 3
rows = n_regions

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

# If only one region, axes won't be 2D — fix that
if n_regions == 1:
    axes = [axes]

for row_idx, region_path in enumerate(region_folders):

    region_name = extract_region_name(os.path.basename(region_path))

    # Pick random task inside region
    tasks = [
        os.path.join(region_path, d)
        for d in os.listdir(region_path)
        if os.path.isdir(os.path.join(region_path, d))
    ]

    if not tasks:
        continue

    task_path = random.choice(tasks)

    # Get PNG files
    png_files = [
        f for f in os.listdir(task_path)
        if f.endswith(".png")
    ]

    if len(png_files) < 3:
        continue

    selected_files = random.sample(png_files, 3)

    for col_idx, file in enumerate(selected_files):

        img = mpimg.imread(os.path.join(task_path, file))
        biomarker_id = extract_biomarker_id(file)

        ax = axes[row_idx][col_idx] if n_regions > 1 else axes[col_idx]
        ax.imshow(img)
        ax.set_title(f"{region_name} | {biomarker_id}", fontsize=10)
        ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()

print(f"Saved grid to {OUTPUT_FILE}")
