'''
Qualitative Evaluation of Monotonicity Constraint in the Stochastic Variational Deep Kernel Model
'''
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import pickle
import sys, os 
from os.path import exists
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from operator import add
import argparse

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

rois_list = ['Amygdala_L', 'Amygdala_R', 'Hippocampus_L', 'Hippocampus_R', 'Lateral_Ventricle_R', 'PHG_R', 'Thalamus_Proper_R']
ROIs= [47, 59, 51, 48, 31, 32, 170]
roi_idxs = [13, 23, 17, 14, 4,  5, 109]
datasets = '1adniblsa'
longitudinal_covariates = pd.read_csv('/home/cbica/Desktop/LongGPClustering/data' + str(2) + '/longitudinal_covariates_subjectsamples_longclean_hmuse_convs_' + 'adniblsa' +'.csv')
longitudinal_covariates['Diagnosis'].replace([-1.0, 0.0, 1.0, 2.0], ['UKN', 'CN', 'MCI', 'AD'], inplace=True)

print('===========Evaluate Performance Metrics=========')
df = {'roi': [], 'lambda': [], 'mse': [], 'mae': [], 'r2': [], 'coverage': [], 'interval': [], 'fold': []}
df_metrics_per_subjects = {'roi': [], 'lambda': [], 'mse': [], 'mae': [], 'roc_dev': [], 'gt_roc': [], 'pred_roc':[], 'fold': [], 'interval': [], 'coverage': []}
for i, r in enumerate(rois_list):

    temporal_dkgp_0 = pd.read_csv('svdk_monotonic_0.0_population_fold_metrics.csv')
    temporal_dkgp_5 = pd.read_csv('svdk_monotonic_5.0_population_fold_metrics.csv')
    temporal_dkgp_10 = pd.read_csv('svdk_monotonic_10.0_population_fold_metrics.csv')

    print(temporal_dkgp_0.columns)

    df['roi'].extend([r for i in range(temporal_dkgp_0.shape[0])])
    df['fold'].extend(temporal_dkgp_0['fold'])
    df['lambda'].extend(['0' for i in range(temporal_dkgp_0.shape[0])])
    df['mae'].extend(temporal_dkgp_0['mae'])
    df['mse'].extend(temporal_dkgp_0['mse'])
    df['r2'].extend(temporal_dkgp_0['r2'])
    df['interval'].extend(temporal_dkgp_0['interval'])
    df['coverage'].extend(temporal_dkgp_0['coverage'])

    df['roi'].extend([r for i in range(temporal_dkgp_5.shape[0])])
    df['fold'].extend(temporal_dkgp_5['fold'])
    df['lambda'].extend(['5' for i in range(temporal_dkgp_5.shape[0])])
    df['mae'].extend(temporal_dkgp_5['mae'])
    df['mse'].extend(temporal_dkgp_5['mse'])
    df['r2'].extend(temporal_dkgp_5['r2'])
    df['interval'].extend(temporal_dkgp_5['interval'])
    df['coverage'].extend(temporal_dkgp_5['coverage'])

    df['roi'].extend([r for i in range(temporal_dkgp_10.shape[0])])
    df['fold'].extend(temporal_dkgp_10['fold'])
    df['lambda'].extend(['10' for i in range(temporal_dkgp_10.shape[0])])
    df['mae'].extend(temporal_dkgp_10['mae'])
    df['mse'].extend(temporal_dkgp_10['mse'])
    df['r2'].extend(temporal_dkgp_10['r2'])
    df['interval'].extend(temporal_dkgp_10['interval'])
    df['coverage'].extend(temporal_dkgp_10['coverage'])

    # load the metrics per subject
    temporal_dkgp_0 = pd.read_csv('svdk_monotonic_0.0_population_per_subject_metrics.csv')
    temporal_dkgp_10 = pd.read_csv('svdk_monotonic_10.0_population_per_subject_metrics.csv')
    temporal_dkgp_5 = pd.read_csv('svdk_monotonic_5.0_population_per_subject_metrics.csv')

    df_metrics_per_subjects['roi'].extend([r for i in range(temporal_dkgp_0.shape[0])])
    df_metrics_per_subjects['fold'].extend(temporal_dkgp_0['fold'])
    df_metrics_per_subjects['lambda'].extend(['0' for i in range(temporal_dkgp_0.shape[0])])
    df_metrics_per_subjects['mae'].extend(temporal_dkgp_0['mae'])
    df_metrics_per_subjects['mse'].extend(temporal_dkgp_0['mse'])
    df_metrics_per_subjects['roc_dev'].extend(temporal_dkgp_0['roc_dev'])
    df_metrics_per_subjects['gt_roc'].extend(temporal_dkgp_0['gt_roc'])
    df_metrics_per_subjects['pred_roc'].extend(temporal_dkgp_0['pred_roc'])
    df_metrics_per_subjects['interval'].extend(temporal_dkgp_0['interval'])
    df_metrics_per_subjects['coverage'].extend(temporal_dkgp_0['coverage'])

    df_metrics_per_subjects['roi'].extend([r for i in range(temporal_dkgp_10.shape[0])])
    df_metrics_per_subjects['fold'].extend(temporal_dkgp_10['fold'])
    df_metrics_per_subjects['lambda'].extend(['10' for i in range(temporal_dkgp_10.shape[0])])
    df_metrics_per_subjects['mae'].extend(temporal_dkgp_10['mae'])
    df_metrics_per_subjects['mse'].extend(temporal_dkgp_10['mse'])
    df_metrics_per_subjects['roc_dev'].extend(temporal_dkgp_10['roc_dev'])
    df_metrics_per_subjects['gt_roc'].extend(temporal_dkgp_10['gt_roc'])
    df_metrics_per_subjects['pred_roc'].extend(temporal_dkgp_10['pred_roc'])
    df_metrics_per_subjects['interval'].extend(temporal_dkgp_10['interval'])
    df_metrics_per_subjects['coverage'].extend(temporal_dkgp_10['coverage'])

    df_metrics_per_subjects['roi'].extend([r for i in range(temporal_dkgp_5.shape[0])])
    df_metrics_per_subjects['fold'].extend(temporal_dkgp_5['fold'])
    df_metrics_per_subjects['lambda'].extend(['5' for i in range(temporal_dkgp_5.shape[0])])
    df_metrics_per_subjects['mae'].extend(temporal_dkgp_5['mae'])
    df_metrics_per_subjects['mse'].extend(temporal_dkgp_5['mse'])
    df_metrics_per_subjects['roc_dev'].extend(temporal_dkgp_5['roc_dev'])
    df_metrics_per_subjects['gt_roc'].extend(temporal_dkgp_5['gt_roc'])
    df_metrics_per_subjects['pred_roc'].extend(temporal_dkgp_5['pred_roc'])
    df_metrics_per_subjects['interval'].extend(temporal_dkgp_5['interval'])
    df_metrics_per_subjects['coverage'].extend(temporal_dkgp_5['coverage'])

df_metrics_per_subjects = pd.DataFrame(data=df_metrics_per_subjects)
df_metrics_per_subjects.to_csv('svdk_monotonicity_performance_metrics_per_subject.csv')
df = pd.DataFrame(data=df)
df.to_csv('svdk_monotonicity_performance_metrics_acrossfolds.csv')

print('Evaluate the Predictive Performance across different rates of monotonicity lambdas')
file_path = 'svdk_monotonicity_performance_metrics_per_subject.csv'
df = pd.read_csv(file_path)

# Calculate mean and standard deviation of MAE for each lambda
mae_stats = df.groupby('lambda')['mae'].agg(['mean', 'std']).reset_index()

# Plot the mean MAE and standard deviation for each lambda as a categorical variable
plt.figure(figsize=(10, 6))
plt.bar(mae_stats['lambda'].astype(str), mae_stats['mean'], yerr=mae_stats['std'], capsize=5)
plt.title('Mean MAE and Standard Deviation for Each Lambda')
plt.xlabel('Lambda')
plt.ylabel('Mean MAE')
plt.grid(axis='y')
plt.savefig('mean_mae_std_of_mae_per_subject.png')
plt.savefig('mean_mae_std_of_mae_per_subject.svg')

# Calculate mean and standard deviation of 'interval' and 'coverage' for each lambda
interval_stats = df.groupby('lambda')['interval'].agg(['mean', 'std']).reset_index()
coverage_stats = df.groupby('lambda')['coverage'].agg(['mean', 'std']).reset_index()

# Plot for 'interval'
plt.figure(figsize=(10, 6))
plt.bar(interval_stats['lambda'].astype(str), interval_stats['mean'], yerr=interval_stats['std'], capsize=5)
plt.title('Mean Interval and Standard Deviation for Each Lambda (Categorical)')
plt.xlabel('Lambda')
plt.ylabel('Mean Interval')
plt.grid(axis='y')
plt.savefig('mean_interval_std_of_interval_per_subject.png')

# Plot for 'coverage'
plt.figure(figsize=(10, 6))
plt.bar(coverage_stats['lambda'].astype(str), coverage_stats['mean'], yerr=coverage_stats['std'], capsize=5)
plt.title('Mean Coverage and Standard Deviation for Each Lambda (Categorical)')
plt.xlabel('Lambda')
plt.ylabel('Mean Coverage')
plt.grid(axis='y')
plt.savefig('mean_coverage_std_of_coverage_per_subject.png')

print('===========EXAMPLES WITH REGRESSION=========')
resultdir = '/home/cbica/Desktop/LongGPClustering'
parser = argparse.ArgumentParser(description='Updated Plots')
parser.add_argument("--datasets", help="GPUs", default='1adniblsa')

args = parser.parse_args()
datasets = args.datasets

roi_names = [ 'Hippocampus R','Thalamus Proper R', 'Lateral Ventricle R', 'Hippocampus L', 'Amygdala R', 'Amygdala L']
roi_idxs = [13, 23, 17, 14, 4,  5]
rois_numbers = [47, 59, 51, 48, 31, 32]

### Trajectories 
for i, r in enumerate(roi_names): 

    if i == 0: 
        showlegend= False
    else: 
        showlegend=False

    print('Produce Samples for', r)
    temporal_dkgp_0 = pd.read_csv('svdk_monotonic_0.0_population_results.csv')
    temporal_dkgp_5 = pd.read_csv('svdk_monotonic_5.0_population_results.csv')
    temporal_dkgp_10 = pd.read_csv('svdk_monotonic_10.0_population_results.csv')

    subjects = list(temporal_dkgp_0['id'].unique())
    counter = 0 
    print('Subjects', len(subjects))

    missing_ids = []
    for s in subjects :
        # s = subjects[o]
        print('Subject', s)
  
        tempdkgp_s_0 = temporal_dkgp_0[temporal_dkgp_0['id'] == s]
        tempdkgp_s_10 = temporal_dkgp_10[temporal_dkgp_10['id'] == s]
        tempdkgp_s_5 = temporal_dkgp_5[temporal_dkgp_5['id'] == s]

        if tempdkgp_s_0.shape[0] >= 5: 


            df = {'value': [], 'model': [], 'time': [], 'upper': [], 'lower': [] }
            # lmm_df = {'value': [], 'time': [], 'upper': [], 'lower': []}
            groundtruth_df = {'value': [], 'time': []}

            longitudinal_covs_u = longitudinal_covariates[longitudinal_covariates['PTID'] == s]
            diagnosis = longitudinal_covs_u['Diagnosis'].tolist()
            age = longitudinal_covs_u['Age'].tolist() 
            baseline_age = age[0]

            tempdkgp_s_0 = tempdkgp_s_0[tempdkgp_s_0['id'] == s]
            tempdkgp_s_10 = tempdkgp_s_10[tempdkgp_s_10['id'] == s]
            tempdkgp_s_5 = tempdkgp_s_5[tempdkgp_s_5['id'] == s]

            groundtruth_df['value'].extend(tempdkgp_s_0['y'])
            groundtruth_df['time'].extend(tempdkgp_s_0['time'].to_list())

            df['model'].extend([0 for i in range(len(tempdkgp_s_0['score'].to_list()))])
            df['value'].extend(tempdkgp_s_0['score'])
            df['time'].extend(tempdkgp_s_0['time'].to_list())
    
            variance = tempdkgp_s_0['variance']
            std = np.sqrt(variance)
            upper = tempdkgp_s_0['score'] + 1.96 * std
            lower = tempdkgp_s_0['score'] - 1.96 * std
            df['upper'].extend(upper)
            df['lower'].extend(lower)

            df['model'].extend([10 for i in range(len(tempdkgp_s_10['score'].to_list()))])
            df['value'].extend(tempdkgp_s_10['score'])
            df['time'].extend(tempdkgp_s_10['time'].to_list())
            variance = tempdkgp_s_10['variance']
            std = np.sqrt(variance)
            upper = tempdkgp_s_10['score'] + 1.96 * std
            lower = tempdkgp_s_10['score'] - 1.96 * std
            df['upper'].extend(upper)
            df['lower'].extend(lower)

            df['model'].extend([5 for i in range(len(tempdkgp_s_5['score'].to_list()))])
            df['value'].extend(tempdkgp_s_5['score'])
            df['time'].extend(tempdkgp_s_5['time'].to_list())
            variance = tempdkgp_s_5['variance']
            std = np.sqrt(variance)
            upper = tempdkgp_s_5['score'] + 1.96 * std
            lower = tempdkgp_s_5['score'] - 1.96 * std
            df['upper'].extend(upper)
            df['lower'].extend(lower)

            df  = pd.DataFrame(data=df)
            # lmm_df = pd.DataFrame(data=)
            groundtruth_df = pd.DataFrame(data=groundtruth_df)

            print(df.shape, groundtruth_df.shape)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=groundtruth_df['time'], 
                y=groundtruth_df['value'], 
                mode='markers+text', 
                name='Noisy Estimates',
                marker=dict(color='black'),  # Light gray
                showlegend=showlegend, 
                text=diagnosis, 
                textposition='top center', 
                textfont=dict(  # Adjust the font properties here
                size=8,  # Font size
                color='black',  # Font color
                family='Arial'  # Font family

            )))

            # Line Plot for Noisy Estimates
            fig.add_trace(go.Scatter(
                x=groundtruth_df['time'], 
                y=groundtruth_df['value'], 
                mode='lines', 
                name='Noisy Estimates',
                showlegend=False,
                line=dict(color='black')
            ))

            # TempDKGP lambda = 0 
            df_0 = df[df['model'] == 0]
            fig.add_trace(go.Scatter(
                x=df_0['time'], 
                y=df_0['value'], 
                mode='lines', 
                showlegend=showlegend, 
                name='TempDKGP Lambda = 0',
                line=dict(color='blue')
            ))

            # TempDKGP CI (change color to distinguish from conformalized interval)
            fig.add_trace(go.Scatter(
                x=df_0['time'].tolist() + df_0['time'].iloc[::-1].tolist(),  # X values: forward time + reverse time
                y=df_0['upper'].tolist() + df_0['lower'].iloc[::-1].tolist(),  # Y values: upper + reversed lower
                fill='toself',
                fillcolor='rgba(0, 114, 178, 0.3)',  # Change this color (e.g., Cornflower blue)
                line=dict(color='rgba(0, 114, 178, 0)', width=0),
                name='TempDKGP CI',
                showlegend=False
            ))
           
            # TempDKGP lambda = 5
            df_5 = df[df['model'] == 5]
            fig.add_trace(go.Scatter(
                x=df_5['time'], 
                y=df_5['value'], 
                mode='lines', 
                showlegend=showlegend, 
                name='TempDKGP Lambda_Mono = 5',
                line=dict(color='orange')
            ))

            # TempDKGP CI (change color to distinguish from conformalized interval)
            fig.add_trace(go.Scatter(
                x=df_5['time'].tolist() + df_5['time'].iloc[::-1].tolist(),  # X values: forward time + reverse time
                y=df_5['upper'].tolist() + df_5['lower'].iloc[::-1].tolist(),  # Y values: upper + reversed lower
                fill='toself',
                fillcolor='rgba(230, 159, 0, 0.3)',  # Change this color (e.g., Cornflower blue)
                line=dict(color='rgba(230, 159, 0, 0)', width=0),
                name='TempDKGP CI',
                showlegend=False
            ))

            # TempDKGP lambda = 10
            df_10 = df[df['model'] == 10]
            fig.add_trace(go.Scatter
            (
                x=df_10['time'], 
                y=df_10['value'], 
                mode='lines', 
                showlegend=showlegend, 
                name='TempDKGP Lambda_Mono = 10',
                line=dict(color='green')
            ))

            # TempDKGP CI (change color to distinguish from conformalized interval)
            fig.add_trace(go.Scatter
            (
                x=df_10['time'].tolist() + df_10['time'].iloc[::-1].tolist(),  # X values: forward time + reverse time
                y=df_10['upper'].tolist() + df_10['lower'].iloc[::-1].tolist(),  # Y values: upper + reversed lower
                fill='toself',
                fillcolor='rgba(0, 158, 115, 0.3)',  # Change this color (e.g., Cornflower blue)
                line=dict(color='rgba(0, 158, 115, 0)', width=0),
                name='TempDKGP CI',
                showlegend=False
            ))

            fig.update_layout(
                title={'text': 'Test Subject at Age ' + "{:.1f}".format(baseline_age), 
                    'y':0.85,
                    'x':0.40,
                    'xanchor': 'center',
                    'yanchor': 'top'}, 
                xaxis_title='Time from baseline (in months)', 
                yaxis_title=r, 
                plot_bgcolor='white',
                width=700, 
                height=400,
                font=dict(
                family="Arial, sans-serif",
                size=20,  # Increases font size for general text
                color="black"
                ),
                paper_bgcolor='white', 
                legend_title_text="ROIs",
                xaxis_showgrid=True,
                yaxis_showgrid=True, 
                xaxis_visible=True, 
                yaxis_visible=True 
            )

            fig.write_image('./results/Monotonicity_predictions_' + s + '_' + r + '_allstudies.png')
            fig.write_image('./results/Monotonicity_predictions_' + s + '_'+ r +'_allstudies.svg')
