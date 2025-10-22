import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def save_bar_plots(base_path: str, output_path: str, biomarkers: list, penalties: list):
    """
    Load metrics CSVs and save bar plots for each metric comparing penalties across biomarkers.

    Supports:
    - If base_path == 'MUSE': multiple biomarkers folders, loop over biomarkers.
    - Else: only one biomarker folder (biomarkers list should contain one element).

    Args:
        base_path (str): Directory where CSV files are located.
        output_path (str): Directory to save the plots.
        biomarkers (list): List of biomarker IDs or a single biomarker folder name.
        penalties (list): List of monotonicity penalties.
    """
    all_metrics = []
    os.makedirs(output_path, exist_ok=True)

    if base_path == 'MUSE':
        # Multiple biomarkers case
        for biomarker in biomarkers:
            for penalty in penalties:
                file_path = os.path.join(base_path, f'H_MUSE_Volume_{biomarker}', f'svdk_monotonic_{penalty}_population_fold_metrics.csv')
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df['biomarker'] = str(biomarker)
                    df['penalty'] = float(penalty)
                    all_metrics.append(df.iloc[0])  # Only row 0 matters
                else:
                    print(f"[!] File not found: {file_path}")
    else:
        # Single biomarker folder case
        biomarker = biomarkers[0]  # Only one biomarker folder expected
        for penalty in penalties:
            file_path = os.path.join(str(biomarker), f'svdk_monotonic_{penalty}_population_fold_metrics.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['biomarker'] = str(biomarker)
                df['penalty'] = float(penalty)
                all_metrics.append(df.iloc[0])  # Only row 0 matters
            else:
                print(f"[!] File not found: {file_path}")

    if not all_metrics:
        print("[!] No data loaded. Please check your file paths.")
        return

    # Create main DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df['penalty'] = metrics_df['penalty'].astype(float)
    metrics_df['biomarker'] = metrics_df['biomarker'].astype(str)

    # Metrics to plot
    metrics_to_plot = ['mae', 'mse', 'r2', 'coverage', 'interval', 'mean_roc_dev']

    sns.set(style='whitegrid')

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=metrics_df,
            x='penalty',
            y=metric,
            hue='biomarker',
            palette='tab10'
        )
        plt.title(f'{metric.upper()} vs Monotonic Penalty')
        plt.xlabel('Monotonic Penalty')
        plt.ylabel(metric.upper())
        plt.legend(title='Biomarker')
        plt.tight_layout()

        # Save plot
        filename = f'{metric}_vs_penalty_bar.png'
        save_path = os.path.join(output_path, filename)
        plt.savefig(save_path)
        plt.close()
        print(f'[✓] Saved: {save_path}')

def parse_monotonicity_txt(base_path: str, biomarkers: list, output_path: str):
    """
    Parse monotonicity percentages from results.txt files and save a bar plot.
    
    Args:
        base_path (str): Base directory containing biomarker folders.
        biomarkers (list): List of biomarker IDs.
        output_path (str): Directory to save the plot.
    """
    records = []

    for biomarker in biomarkers:
        results_file = os.path.join(base_path, f'H_MUSE_Volume_{biomarker}', 'results.txt')
        if not os.path.exists(results_file):
            print(f"[!] results.txt not found: {results_file}")
            continue
        
        with open(results_file, 'r') as f:
            content = f.read()
        
        # Regex to extract percentage and penalty
        pattern = r'Percentage of samples where monotonicity is achieved:\s*([\d\.]+)% for lambda_penalty\s*([\d\.]+)'
        matches = re.findall(pattern, content)
        
        if not matches:
            print(f"[!] No monotonicity data found in {results_file}")
            continue
        
        for percentage_str, penalty_str in matches:
            records.append({
                'biomarker': str(biomarker),
                'penalty': float(penalty_str),
                'monotonicity': float(percentage_str)
            })

    if not records:
        print("[!] No monotonicity data parsed. Check files.")
        return
    
    df = pd.DataFrame(records)
    
    # Plot bar plot
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='penalty', y='monotonicity', hue='biomarker', palette='tab10')
    plt.title('Monotonicity Percentage vs Monotonic Penalty')
    plt.xlabel('Monotonic Penalty')
    plt.ylabel('Percentage of Samples with Monotonicity (%)')
    plt.legend(title='Biomarker')
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, 'monotonicity_vs_penalty_bar.png')
    plt.savefig(save_path)
    plt.close()
    print(f'[✓] Saved monotonicity plot: {save_path}')


save_bar_plots(
    base_path='MUSE',
    output_path='./plots/MUSE/penalty_comparison',
    biomarkers=[47, 48, 51, 52],
    penalties=[0.0, 0.75, 1.0, 5.0]
)

save_bar_plots(
    base_path='SPARE_AD',
    output_path='./plots/SPARE_AD',
    biomarkers=['SPARE_AD'],  # single element list
    penalties=[0.0, 0.75, 1.0, 5.0]
)

save_bar_plots(
    base_path='SPARE_BA',
    output_path='./plots/SPARE_BA',
    biomarkers=['SPARE_BA'],  # single element list
    penalties=[0.0, 0.75, 1.0, 5.0]
)

save_bar_plots(
    base_path='ADAS',
    output_path='./plots/ADAS',
    biomarkers=['ADAS'],  # single element list
    penalties=[0.0, 0.75, 1.0, 5.0]
)

save_bar_plots(
    base_path='MMSE',
    output_path='./plots/MMSE',
    biomarkers=['MMSE'],  # single element list
    penalties=[0.0, 0.75, 1.0, 5.0]
)