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
parser.add_argument("--file", help="Identifier for the data", default="subjectsamples_longclean_hmuse_adniblsa")
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
# Clinical_Features = Diagnosis, Age, Sex, APOE4 Alleles, Education Years
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

datasamples = pd.read_csv('/home/cbica/Desktop/LongGPClustering/data'+str(folder)+'/' + file + '.csv')
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
with (open("/home/cbica/Desktop/LongGPClustering/data"+str(folder)+"/train_subject_"+expID+"_ids_hmuse" + "" + str(fold) +  ".pkl", "rb")) as openfile:
    while True:
        try:
            train_ids.append(pickle.load(openfile))
        except EOFError:
            break 
    
with (open("/home/cbica/Desktop/LongGPClustering/data"+str(folder)+"/test_subject_"+expID+"_ids_hmuse" + "" + str(fold) + ".pkl", "rb")) as openfile:
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

datasamples = pd.read_csv('/home/cbica/Desktop/LongGPClustering/data'+str(folder)+'/' + file + '.csv')
### SET UP THE TRAIN/TEST/VAL DATA## 
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

single_muse = 'H_MUSE_Volume_47'
print('ROI NAME', single_muse )

if single_muse == 'SPARE_AD': 
    list_index = 0 
elif single_muse == 'SPARE_BA': 
    list_index = 1 
else: 
    print('MUSE ROI', single_muse)
    list_index = roi_to_idx[single_muse.split('_')[-1]]

train_y = train_y_all[:, list_index]
test_y = test_y_all[:, list_index]
val_y = val_y[:, list_index]

# Define the parameter grid for the MLP
param_grid = {
    'hidden_layer_sizes': [(128, 64) ],  # More layer configurations
    'activation': ['relu'],  # Different activation functions
    'solver': ['adam'],  # Different solvers
    'alpha': [0.001, 0.01],  # Regularization parameter
    'learning_rate': ['constant'],  # Learning rate schedule
    'learning_rate_init': [0.01],  # Initial learning rate
    'max_iter': [200],  # Number of epochs
    'early_stopping': [True]  # Enable early stopping
}

# Initialize the MLPRegressor
mlp = MLPRegressor(random_state=42)
# Create a custom scorer based on mean squared error
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring=scorer, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the training data
grid_search.fit(train_x, train_y)

# Print out the best parameters and the corresponding score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

# Use the best model to predict the test set
best_mlp = grid_search.best_estimator_

best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Print out the best parameters and the corresponding score
print(f"Best parameters found: {best_params}")
print(f"Best cross-validation score: {best_score}")

y_pred = best_mlp.predict(test_x)

# Evaluate the best model with the test data
mse_ = mean_squared_error(test_y, y_pred)
mae_ = mean_absolute_error(test_y, y_pred)
rmse = np.sqrt(mse_)
rsq = r2_score(test_y, y_pred)

print(f'Mean Squared Error: {mse_}')
print(f'Mean Absolute Error: {mae_}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {rsq}')

# Store the best parameters and score in a JSON file for future reference
best_results = {
    "best_params": best_params,
    "best_score": best_score, 
    "mse": float(mse_),    # Convert to native float
    "mae": float(mae_),    # Convert to native float
    "rmse": float(rmse),   # Convert to native float
    "rsq": float(rsq)      # Convert to native float
}

# Save the dictionary to a JSON file
with open("./results/deep_kernel_design_results_1.json", "w") as json_file:
    json.dump(best_results, json_file)



## Store the Weights of the best MLP Regressor ###

# Extract the weights and biases
coefs = best_mlp.coefs_  # List of weight matrices
intercepts = best_mlp.intercepts_  # List of bias vectors

# Store the weights and biases in a file
with open('best_mlp_weights.pkl', 'wb') as f:
    pickle.dump({'coefs': coefs, 'intercepts': intercepts}, f)

print("Weights and biases stored successfully!") 
transfer_weights = True
if transfer_weights: 
    print('Transfer Sklearn Weights and do inference on test data')
    import torch
    import torch.nn as nn
    import pickle

    # class MLP(nn.Module):
    #     def __init__(self):
    #         super(MLP, self).__init__()
    #         self.fc1 = nn.Linear(151, 256)
    #         self.relu1 = nn.ReLU()
    #         self.fc2 = nn.Linear(256, 128)
    #         self.relu2 = nn.ReLU()
    #         self.fc3 = nn.Linear(128, 64)
    #         self.relu3 = nn.ReLU()
    #         self.fc4 = nn.Linear(64, 32)
    #         self.relu4 = nn.ReLU()
    #         self.fc5 = nn.Linear(32, 64)
    #         self.relu5 = nn.ReLU()
    #         self.output = nn.Linear(64, 1)

    #     def forward(self, x):
    #         x = self.relu1(self.fc1(x))
    #         x = self.relu2(self.fc2(x))
    #         x = self.relu3(self.fc3(x))
    #         x = self.relu4(self.fc4(x))
    #         x = self.relu5(self.fc5(x))
    #         x = self.output(x)
    #         return x

    class MLP_Reduced(nn.Module):
        def __init__(self):
            super(MLP_Reduced, self).__init__()
            self.fc1 = nn.Linear(151, 128)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.relu2 = nn.ReLU()
            self.output = nn.Linear(64, 1)

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.output(x)
            return x


    # Initialize the model
    model = MLP_Reduced()

    # Load the stored weights and biases
    with open('best_mlp_weights.pkl', 'rb') as f:
        params = pickle.load(f)

    coefs = params['coefs']
    intercepts = params['intercepts']
    
    for i in range(len(coefs)):
        print(f'Layer {i + 1} weights shape: {coefs[i].shape}')
        print(f'Layer {i + 1} biases shape: {intercepts[i].shape}')

    ## Now get the weights of the pytorch model to make sure of the shape they have
    for name, param in model.named_parameters():
        print(name, param.shape)

    # Transfer the weights and biases to the PyTorch model
    model.fc1.weight.data = torch.tensor(coefs[0].T)  # Transpose to match PyTorch's weight shape (That's Verified!)
    model.fc1.bias.data = torch.tensor(intercepts[0])

    model.fc2.weight.data = torch.tensor(coefs[1].T)
    model.fc2.bias.data = torch.tensor(intercepts[1])

    model.fc3.weight.data = torch.tensor(coefs[2].T)
    model.fc3.bias.data = torch.tensor(intercepts[2])

    model.fc4.weight.data = torch.tensor(coefs[3].T)
    model.fc4.bias.data = torch.tensor(intercepts[3])

    model.fc5.weight.data = torch.tensor(coefs[4].T)
    model.fc5.bias.data = torch.tensor(intercepts[4])

    model.output.weight.data = torch.tensor(coefs[2].T)
    model.output.bias.data = torch.tensor(intercepts[2])

    print("PyTorch model initialized with sklearn weights!")

    # Convert the test data to a PyTorch tensor
    test_x = torch.tensor(test_x, dtype=torch.float32)

    # Make predictions with the PyTorch model
    y_pred = model(test_x)


    # Evaluate the best model with the test data
    py_mse_ = mean_squared_error(test_y, y_pred.detach().numpy())
    py_mae_ = mean_absolute_error(test_y, y_pred.detach().numpy())
    py_rmse = np.sqrt(py_mse_)
    py_rsq = r2_score(test_y, y_pred.detach().numpy())

    print(f'Mean Squared Error: {py_mse_}')
    print(f'Mean Absolute Error: {py_mae_}')
    print(f'Root Mean Squared Error: {py_rmse}')
    print(f'R-squared: {py_rsq}')

    # Compare the mse_ and py_mse_ values and If they are very close mark the tranfer of weights as successful
    success = True
    if np.isclose(mse_, py_mse_, atol=1e-5):
        pass
    else:
        success = False

    if np.isclose(mae_, py_mae_, atol=1e-5):
        pass
    else:
        success = False

    if success: 
        print('Weights Transfer Successful !!!')





