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
parser.add_argument("--file", help="Identifier for the data", default="subjectsamples_longclean_spare_adniblsa")
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
parser.add_argument("--task", help='Task id', type=str, default="SPARE")  # Right Hippocampus 
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


if personalization: 
    personalization_results = {'id': [], 'kfold': [], 'score': [], 'y': [], 'time': [], 'ROI': [], 'history_points': [] }
    person_metrics_history = {'history': [], 'ae': [], 'se': [],  'model':[], 'id': [], 'time': [], 'kfold': [], 'ROI': [], 'Tobs': [] }
    person_mean_metrics = {'id': [], 'history': [], 'mae': [], 'mse': [], 'rmse': [], 'model': [], 'kfold': [], 'ROI': [] }

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

# run only a singe fold
for fold in range(5): 
    print('FOLD::', fold)
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

    # these are already scaled
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    ### Select only the ROIS that I already have results for ##

    hmuse_rois_reduced = ['H_MUSE_Volume_47', 'H_MUSE_Volume_59', 'H_MUSE_Volume_51', 'H_MUSE_Volume_31', 'H_MUSE_Volume_32']
    hmuse_rois_reduced.append('H_MUSE_Volume_48')
    hmuse_rois_reduced.append('H_MUSE_Volume_170')

    ROIS_names  = ['Hippocampus R', 'Thalamus Proper R', 'Lateral Ventricle R', 'Hippocampus L', 'Amygdala R', 'Amygdala L', 'PHG R']


    hmuse_rois_reduced = ['SPARE_AD', 'SPARE_BA']


    for ind, roi in enumerate(hmuse_rois_reduced): 

        single_muse = roi
        print('ROI NAME', roi )
        # roi_idx = roi.split('_')[-1]

        # list_index = roi_to_idx[str(roi_idx)]

        if single_muse == 'SPARE_AD': 
            list_index = 0 
        elif single_muse == 'SPARE_BA': 
            list_index = 1 
        else: 
            print('MUSE ROI', single_muse)
            list_index = roi_to_idx[single_muse.split('_')[-1]]
            print(list_index)

        print(list_index)
        train_y = train_y_all[:, list_index]
        test_y = test_y_all[:, list_index]

        # Select a single ROI 
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        # Initialize the MLP with the specified parameters
        print('Train the MLP Population Model')
        # Train a general population model
        # Train a general population model
        mlp_population = MLPRegressor(
            hidden_layer_sizes=(100, 50, 100),
            activation='relu',
            solver='sgd',
            alpha=0.01,
            learning_rate='adaptive',
            learning_rate_init=0.001844,  # Initial learning rate for population training
            max_iter=200,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10  # Stop training if no improvement after 10 epochs
        )

        print('Train the MLP Population Model', type(train_x),type(train_y))

        contains_nan = np.isnan(train_x).any()

        contains_nan = np.isnan(train_y).any()


        mlp_population.fit(train_x, train_y)  # train on a general dataset

        y_pred = mlp_population.predict(test_x)

        # Evaluate the best grid model with the test data
        print('Test X, Y', test_x.shape, test_y.shape)
        print('Prediction', y_pred.shape)
        mse_ = mean_squared_error(test_y, y_pred)
        print('MSE', mse_.shape, mse_)
        mae_ = mean_absolute_error(test_y, y_pred)
        print('MAE', mae_.shape, mae_)

        rmse = np.sqrt(mse_)
        print('RMSE', rmse.shape, rmse)
        rsq = r2_score(test_y, y_pred)
        print('R2', rsq.shape)

        print(f'Mean Squared Error: {mse_}')
        print(f'R-squared: {rsq}')

        population_results['id'].extend(corresponding_test_ids) 
        population_results['kfold'].extend([fold for c in range(len(corresponding_test_ids))])
        population_results['score'].extend(y_pred.tolist())
        population_results['lower'].extend([0 for k in range(len(y_pred.tolist()))])
        population_results['upper'].extend([0 for k in range(len(y_pred.tolist()))]) 
        population_results['y'].extend(test_y.tolist())
        time_ = test_x[:, -1].tolist() 
        population_results['time'].extend(time_) 
        population_results['ROI'].extend([single_muse for i in range(len(time_))])

        population_mae_kfold['ROI'].append(single_muse)
        population_mae_kfold['mae'].append(mae_)
        population_mae_kfold['mse'].append(mse_)
        population_mae_kfold['rmse'].append(rmse)
        population_mae_kfold['R2'].append(rsq)
        population_mae_kfold['kfold'].append(fold)
        population_mae_kfold['coverage'].append(0)
        population_mae_kfold['interval'].append(0)


        # Create a set of unique subjects in the test set
        unique_subjects = set(corresponding_test_ids)

        # Iterate over each unique subject to compute MAE
        for test_subject in unique_subjects:
            # Find the indices of all occurrences of this subject
            subject_indices = [i for i, x in enumerate(corresponding_test_ids) if x == test_subject]
            
            # Calculate the MAE for this subject
            mae_subject = np.mean(np.abs(np.array(test_y)[subject_indices] - np.array(y_pred)[subject_indices]))
            
            # Store the results
            population_metrics_per_subject['kfold'].append(fold)  # Assuming 'fold' is defined earlier in your code
            population_metrics_per_subject['id'].append(test_subject)
            population_metrics_per_subject['mae_per_subject'].append(mae_subject)
            population_metrics_per_subject['interval'].append(0)  # Placeholder for interval if needed later
            population_metrics_per_subject['coverage'].append(0)  # Placeholder for coverage if needed later
            population_metrics_per_subject['ROI'].append(single_muse)

        print('Population Metrics Per Subject', len(population_metrics_per_subject['id']), len(population_metrics_per_subject['mae_per_subject']))
        # assert 
        assert len(population_metrics_per_subject['id']) == len(population_metrics_per_subject['mae_per_subject'])

        personalization = False 
        if personalization: 
            ## Keep the Deep Kernel Stable and train a new GP with the History of the Subject 
            print('START THE PERSONALIZATION FOR THE MLP REGRESSION')
            test_ids = list(set(corresponding_test_ids))
            pers_mae, pers_interval, pers_coverage = [], [], [] 
            # personalize only 1 subjevct so as to check the error.
            for test_subject in test_ids:  

                # Data # 
                test_subject_x = datasamples[datasamples['PTID'].isin([test_subject])]['X']
                test_subject_y = datasamples[datasamples['PTID'].isin([test_subject])]['Y']
            
                if test_subject_x.shape[0] > 7: 
                    print('Test Subject', test_subject)
                    longitudinal_covariates_subject = longitudinal_covariates[longitudinal_covariates['PTID'].isin([test_subject])]

                    test_subject_x, test_subject_y, _, _ = process_temporal_singletask_data(train_x=test_subject_x, train_y=test_subject_y, test_x=test_subject_x, test_y=test_subject_y, test_ids=test_ids)

                    test_subject_y = test_subject_y[:, list_index]
                    test_subject_y = test_subject_y.squeeze()

                    for h in [2,3,4,5,6,7]: 

                        # print('Test X, Y', test_x.shape, test_y.shape)
                        train_x_subject = test_subject_x[:h, :]
                        train_y_subject = test_subject_y[:h]

                        test_sub_x =  test_subject_x[h:, :]
                        test_sub_y =  test_subject_y[h:]
                        
                        new_train = np.concatenate((train_x, train_x_subject), axis=0)
                        new_targets = np.concatenate((train_y, train_y_subject), axis=0)
                        
                        #### Personalization with Warm Start:You can also use the warm_start feature of MLPRegressor
                        #  to fine-tune the model with the existing weights. 
                        # This method enables incremental learning where you can continue training from where the model left off 
                        # but with different hyperparameters for a specific subset of training iterations.
                        # Initialize the personalized model with the specific learning rate and epochs
                        
                        mlp_subject = MLPRegressor(
                            hidden_layer_sizes=(100, 50, 100),
                            activation='relu',
                            solver='sgd',
                            alpha=0.01,
                            learning_rate='adaptive',
                            learning_rate_init=0.0001,  # Personalized learning rate
                            max_iter=1,  # Set to 1 for iterative training
                            warm_start=True,  # Enable warm start
                            random_state=42
                        )
                        # Fit the model with a minimal subset to initialize internal configurations
                        mlp_subject.fit(train_x_subject[:1], train_y_subject[:1])  # Use only a minimal subset


                        # Use the weights from the population model to initialize the subject-specific model
                        mlp_subject.coefs_ = [np.array(coef) for coef in mlp_population.coefs_]
                        mlp_subject.intercepts_ = [np.array(intercept) for intercept in mlp_population.intercepts_]
                        
                        # Continue training with actual subject data
                        mlp_subject.max_iter = 50  # Set desired number of epochs for fine-tuning
                        mlp_subject.fit(train_x_subject, train_y_subject)

                        y_pred_sub = mlp_subject.predict(test_sub_x)

                        mae_pers, ae_pers = mae(test_sub_y, y_pred_sub)
                        mse_pers, rmse_pers, se_pers = mse(test_sub_y, y_pred_sub)  

                        all_subject_predictions = mlp_subject.predict(test_subject_x)

                        Tobs = test_subject_x[:,-1].tolist()[h-1]
                        time_ = test_sub_x[:, -1].tolist() # this is the test_time 

                        ## Store the Evaluation Metrics on Unseen Trajectories
                        person_metrics_history['id'].extend([test_subject for o in range(test_sub_x.shape[0])])
                        person_metrics_history['time'].extend(time_)
                        person_metrics_history['history'].extend([h for p in range(test_sub_x.shape[0])])
                        person_metrics_history['Tobs'].extend([Tobs for t in range(test_sub_x.shape[0])])
                        person_metrics_history['ae'].extend(ae_pers.tolist())
                        person_metrics_history['se'].extend(se_pers.tolist())
                        person_metrics_history['model'].extend(['Personalized' for p in range(test_sub_x.shape[0])])
                        person_metrics_history['kfold'].extend([fold for p in range(test_sub_x.shape[0])])
                        person_metrics_history['ROI'].extend([ roi for p in range(test_sub_x.shape[0])])

                        ## Store the Mean Evaluation Metrics on Unseen Trajectories
                        person_mean_metrics['id'].append(test_subject)
                        person_mean_metrics['history'].append(h)
                        person_mean_metrics['mse'].append(mse_pers)
                        person_mean_metrics['mae'].append(mae_pers)
                        person_mean_metrics['rmse'].append(rmse_pers)
                        person_mean_metrics['model'].append('Personalized')
                        person_mean_metrics['kfold'].append(fold)
                        person_mean_metrics['ROI'].append(roi)

                        ## Store the Whole Trajectory 
                        whole_time_ = test_subject_x[:, -1].tolist() # this is the time in the whole trajectory
                        personalization_results['id'].extend([test_subject for o in range(test_subject_x.shape[0])])
                        personalization_results['kfold'].extend([fold for o in range(test_subject_x.shape[0])])
                        personalization_results['score'].extend(all_subject_predictions)
                        personalization_results['y'].extend(test_subject_y.tolist())
                        personalization_results['time'].extend(whole_time_) 
                        personalization_results['ROI'].extend([roi for i  in range(test_subject_x.shape[0])])
                        Tobs = test_subject_x[:,-1].tolist()[h-1]
                        personalization_results['history_points'].extend([h for i in range(len(test_subject_x.tolist()))])

                           
if personalization:

    for k in personalization_results.keys(): 
        print(k, len(personalization_results[k]))

    for k in person_metrics_history.keys(): 
        print(k, len(person_metrics_history[k]))

    for k in person_mean_metrics.keys(): 
        print(k, len(person_mean_metrics[k])) 

    personalization_results_df = pd.DataFrame(data=personalization_results)
    personalization_results_df.to_csv('./neuripsresults/person_mlpsk_singletask_' + str(task) + '_dkgp_population_'+ expID+'.csv')

    person_metrics_history_df = pd.DataFrame(data=person_metrics_history)
    person_mean_metrics_df = pd.DataFrame(data=person_mean_metrics) 

    person_metrics_history_df.to_csv('./neuripsresults/person_mlpsk_metrics_results_dkgp_'+ str(task) + '_' + expID +'.csv')
    person_mean_metrics_df.to_csv('./neuripsresults/person_mlpsk_mean_metrics_dkgp_' + str(task) + '_' + expID +'.csv')

### Store the population results ###
results_df = pd.DataFrame(population_results)
mae_df = pd.DataFrame(population_mae_kfold)
metrics_per_subject = pd.DataFrame(population_metrics_per_subject)

results_df.to_csv('/home/cbica/Desktop/LongGPClustering/baselineonlyresults/mlpsk_'+str(task)+'_results_'+expID+'.csv')
mae_df.to_csv('/home/cbica/Desktop/LongGPClustering/baselineonlyresults/mlpsk_'+str(task)+'_mae_'+expID+'.csv')
metrics_per_subject.to_csv('/home/cbica/Desktop/LongGPClustering/baselineonlyresults/mlpsk_'+str(task)+'_metrics_per_subject'+expID+'.csv')