'''
Script for the Multioutput Regression Model: Neurodegeneration and Cognition
'''

### MLP Regression Sklearn ####
import pandas as pd
import numpy as np
from pandas.core.arrays.sparse import dtype
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import argparse
import time
import json
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from functions import process_temporal_singletask_data, mae, mse
from sklearn.neural_network import MLPRegressor
import torch 
from sklearn.base import clone
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import json



sns.set_style("white", {'axes.grid' : False})

# Plot Controls 
# sns.set_theme(context="paper",style="whitegrid", rc={'axes.grid' : True, 'font.serif': 'Times New Roman'})

# wandb.init(project="HMUSEDeepSingleTask", entity="vtassop", save_code=True)
parser = argparse.ArgumentParser(description='Deep Regression for Neurodegeneration Prediction')
## Data Parameters 
parser.add_argument("--gpuid", help="GPUs", default=0)
parser.add_argument("--experimentID", help="Indicates the Experiment Identifier", default='adniblsa') # 1adni normally
parser.add_argument("--exp", help="Indicates the modality", default='')
parser.add_argument("--kfoldID", help="Identifier for the Kfold IDs", default="")
parser.add_argument("--file", help="Identifier for the data", default="subjectsamples_baseline_cognitive_pet_adni")
parser.add_argument("--covariates", help="String with the Covariates used as Condition in the Model", default="Diagnosis-Age-Sex-APOE4_Alleles-Education_Years")
parser.add_argument("--genomic", help="Indicates the genomic modality", default=0, type=int)
parser.add_argument("--followup", help="Indicates the followup information", default=0, type=int)
## Kernel Parameters ##
parser.add_argument('--kernel_choice', help='Indicates the selection of GP kernel', default='RBF', type=str)
parser.add_argument('--mean', help='Selection of Mean function', default='Constant', type=str)
## Deep Kernel Parameters 
parser.add_argument("--depth", help='indicates the depth of the Deep Kernel', default='')
parser.add_argument("--activ", help='indicates the activation function', default='relu')
parser.add_argument("--dropout", help='indicates the dropout rate', default=0.1, type=float)
## Training Parameters
parser.add_argument("--iterations", help="Epochs", default=500)
parser.add_argument("--optimizer", help='Optimizer', default='adam')
parser.add_argument("--learning_rate", help='Learning Rate', type=float, default=0.01844)    # 0.01844 is in hmuse rois 
parser.add_argument("--task", help='Task id', type=str, default="MUSE")  # Right Hippocampus 
parser.add_argument("--roi_idx", type=int, default=-1)
# Personalization # 
parser.add_argument("--personalization", type=str, default=False)
parser.add_argument("--history", type=int, default=4)
parser.add_argument("--pers_lr", type=float, help='Pers LR', default=0.01844) # 0.3510
parser.add_argument("--folder", type=int, default=2)

####
# Feature Representation 
# H_MUSE features
# Clinical_Features = Diagnosis, Age, Sex, APOE4 Alleles, Education Years,
# PET Features 'AV45_SUVR', 'FDG', 'PIB_SUVR', 'PIB_SUVR_PAC', 'PIB_Status', 'PTau_CSF', 'Tau_CSF'
# Cognitive Features 'MMSE_nearest_2.0', 'ADAS_COG_11', 'ADAS_COG_13', 'ADNI_EF', 'ADNI_LAN', 'ADNI_MEM'
#  PET + Cognitive Scores (at baseline)
# Genomic 
# Follow-Up = {HMUSE}{Time}
# Time 
####

t0= time.time()
args = parser.parse_args()
history = int(args.history)
personalization = args.personalization
genomic = args.genomic 
gpu_id = int(args.gpuid) 
exp = args.exp 
iterations = int(args.iterations)
covariates = args.covariates.split("-")
kfoldID = args.kfoldID
file = args.file 
expID = args.experimentID 
genomic = args.genomic 
followup = args.followup
depth = args.depth 
activ = args.activ 
dropout = args.dropout
task = args.task 
kernel = args.kernel_choice
mean = args.mean
roi_idx = args.roi_idx

text_task = task 

personalization = False 
mae_TempGP_list, mae_TempGP_list_comp = [], []  
population_results = {'ROI': [], 'id' : [], 'kfold': [], 'score': [], 'upper': [], 'lower': [], 'y': [], 'time': []}
population_mae_kfold = {'ROI': [], 'kfold': [], 'mae': [], 'mse': [], 'rmse': [], 'R2': [],  'interval': [], 'coverage': []}
population_metrics_per_subject = {'kfold': [], 'id': [], 'mae_per_subject': [], 'interval': [], 'coverage': [], 'ROI': [] }
mae_MTGP_list, coverage_MTGP_list, interval_MTGP_list = [], [], [] 

folder = int(args.folder)
# 2 adniblsa

datasamples = pd.read_csv('./data/' + file + '.csv')
# covariatesdf = pd.read_csv('/home/cbica/Desktop/LongGPClustering/data'+str(folder)+'/covariates_subjectsamples_longclean_hmuse_allstudies.csv')
longitudinal_covariates = pd.read_csv('/home/cbica/Desktop/LongGPClustering/data' + str(folder) + '/longitudinal_covariates_subjectsamples_longclean_hmuse_convs_'+expID+'.csv')
subject_ids = list(datasamples['PTID'].unique()) 

# wandb.config['Subjects'] = len(subject_ids) 

f = open('/home/cbica/Desktop/LongGPClustering/roi_to_idx.json')
roi_to_idx = json.load(f)

print(roi_to_idx)

index_to_roi = {v: k for k, v in roi_to_idx.items()}

print(index_to_roi)

fold = 0
train_ids, test_ids = [], []
with (open("./data/train_subjects_adni" + str(fold) +  ".pkl", "rb")) as openfile:
    while True:
        try:
            train_ids.append(pickle.load(openfile))
        except EOFError:
            break 
    
with (open("./data/test_subjects_adni"+ str(fold) + ".pkl", "rb")) as openfile:
    while True:
        try:
            test_ids.append(pickle.load(openfile))
        except EOFError:
            break


train_ids = train_ids[0]
test_ids = test_ids[0]
val_ids = train_ids[0:10]
train_ids = train_ids[10:]

print('Train IDs', len(train_ids))
print('Test IDs', len(test_ids))
print('Val IDs', len(val_ids))
print() 
# print('Train', train_ids)
# print('Test', test_ids)
for t in test_ids: 
    if t in train_ids: 
        raise ValueError('Test Samples belong to the train!')

### SET UP THE TRAIN/TEST DATA FOR THE MULTITASK GP### 
train_x = datasamples[datasamples['PTID'].isin(train_ids)]['X']
train_y = datasamples[datasamples['PTID'].isin(train_ids)]['Y']
test_x = datasamples[datasamples['PTID'].isin(test_ids)]['X']
test_y = datasamples[datasamples['PTID'].isin(test_ids)]['Y']
val_x = datasamples[datasamples['PTID'].isin(val_ids)]['X']
val_y = datasamples[datasamples['PTID'].isin(val_ids)]['Y']
    
corresponding_test_ids = datasamples[datasamples['PTID'].isin(test_ids)]['PTID'].to_list()
corresponding_train_ids = datasamples[datasamples['PTID'].isin(train_ids)]['PTID'].to_list() 
assert len(corresponding_test_ids) == test_x.shape[0]
assert len(corresponding_train_ids) == train_x.shape[0]

train_x, train_y_all, test_x, test_y_all = process_temporal_singletask_data(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, test_ids=test_ids)
val_x, val_y, val_x, val_y = process_temporal_singletask_data(train_x=val_x, train_y=val_y, test_x=val_x, test_y=val_y, test_ids=test_ids)

print(type(train_x), type(train_y_all), type(test_x), type(test_y_all))

# convert tensor to numpy array 
train_x = train_x.numpy()
train_y_all = train_y_all.numpy()
test_x = test_x.numpy()
test_y_all = test_y_all.numpy()


targets = ['H_MUSE_Volume_47','SPARE_AD', 'ADAS_COG_13', 'MMSE_2.0_nearest']
target_identification = ['Atrophy', 'Index', 'Cognition', 'Cognition']
## The target sequence is: 
## First the 145 ROIs, then the Composite Biomarkers (SPARE-AD, SPARE-BA) and then the Cognitive Scores (ADAS-Cog, MMSE)
## Index for ROIs (0,144)
## Index for Composite Biomarkers (145,146)
## Index for Cognitive Scores (147,148)


index_list = []
for idx, target in enumerate(targets): 
    if target_identification[idx] == 'Atrophy':
        print('Target', target)
        index_list.append(roi_to_idx[target.split('_')[-1]])
    elif target_identification[idx] == 'Index': 
        print('Target', target)
        if  target == 'SPARE_AD':
            index_list.append(145)
        elif target == 'SPARE_BA':
            index_list.append(146)
    elif target_identification[idx] == 'Cognition':
        print('Target', target)
        if target == 'ADAS_COG_13':
            index_list.append(147)
        elif target == 'MMSE_2.0_nearest':
            index_list.append(148)

print('Index List', index_list)

### Keep Only the index_list indices from the train_y_all 
train_y = train_y_all[:, index_list]
test_y = test_y_all[:, index_list]
val_y = val_y[:, index_list]

print(f'Train X Shape: {train_x.shape}, Train Y Shape: {train_y.shape}, Test X Shape: {test_x.shape}, Test Y Shape: {test_y.shape}, Val X Shape: {val_x.shape}, Val Y Shape: {val_y.shape}')


class MultiOutputMLPRegressor(MLPRegressor):
    def predict(self, X):
        return super().predict(X).reshape(X.shape[0], -1)


def custom_multi_metric_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    mse = mean_squared_error(y, y_pred, multioutput='uniform_average')
    mae = mean_absolute_error(y, y_pred, multioutput='uniform_average')
    r2 = r2_score(y, y_pred, multioutput='uniform_average')
    return (r2 - mse - mae) / 3  # A combined score

scorer = custom_multi_metric_scorer

param_grid = {
    'hidden_layer_sizes': [
        (512, 256, 128),  # Gradually decreasing width
        (1024, 512, 256),  # Gradually decreasing width
        (512, 256, 128, 64),  # Gradually decreasing width
        (256, 128, 64, 32),  # Gradually decreasing width
    ],
    'activation': ['relu'],
    'solver': ['adam'],  # Adam is generally good for complex architectures
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['adaptive'],  # Adaptive learning rate can be beneficial for complex problems
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [500, 1000],  # Increased max iterations for more complex models
    'batch_size': [64, 128, 256],  # Larger batch sizes for stability
    'early_stopping': [True],
    'n_iter_no_change': [10, 40],  # Number of iterations with no improvement to wait before early stopping
}

# Set up the GridSearchCV
# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=MultiOutputMLPRegressor(random_state=42, early_stopping=True),
    param_grid=param_grid,
    scoring=custom_multi_metric_scorer,
    cv=5,
    n_jobs=-1,
    verbose=2
)
print("Starting GridSearchCV...")
grid_search.fit(train_x, train_y)
print("GridSearchCV completed.")
# Get best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters found: {best_params}")
print(f"Best cross-validation score: {best_score}")

# Save best parameters and score
best_results = {
    "best_params": best_params,
    "best_score": best_score
}

with open('best_multioutput_mlp_params.json', 'w') as json_file:
    json.dump(best_results, json_file)

# Use best model to predict
best_mlp = grid_search.best_estimator_
y_pred = best_mlp.predict(test_x)

# Evaluate the model
mse = mean_squared_error(test_y, y_pred, multioutput='raw_values')
r2 = r2_score(test_y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(test_y, y_pred, multioutput='raw_values')

print('Evaluation on Test Set:')
for i, t in enumerate(targets):
    print(f"Target: {t}")
    print(f"MSE: {mse[i]}")
    print(f"R2: {r2[i]}")
    print(f"MAE: {mae[i]}")
    print()

