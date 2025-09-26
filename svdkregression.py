# SVDK Regression Model for Population and then We will proceed to Personalization
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import json
import pickle
import tqdm
import torch
import gpytorch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from functions import process_temporal_singletask_data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from models import MLP, GaussianProcessLayer, DKLModel, AdditiveDKLModel, AdditiveGaussianProcessLayer, MLP_Reduced

import matplotlib.pyplot as plt

import gpytorch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from datasets import TrainDataset, TestDataset
from utils import freeze_layers

from torch.optim.lr_scheduler import LambdaLR

def warmup_lambda(epoch):
    if epoch < 10:
        return 0.1 * (epoch + 1)
    return 1.0

def print_variational_params(model):
    var_mean = model.gp_layer.variational_strategy._variational_distribution.variational_mean
    var_cov = model.gp_layer.variational_strategy._variational_distribution.chol_variational_covar
    print(f"Variational mean - Mean: {var_mean.mean().item():.4f}, Std: {var_mean.std().item():.4f}")
    print(f"Variational covar - Mean: {var_cov.mean().item():.4f}, Std: {var_cov.std().item():.4f}")

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 10:
                print(f"Exploding gradient in {name}: {grad_norm}")
            elif grad_norm < 1e-8:
                print(f"Vanishing gradient in {name}: {grad_norm}")

class CustomVariationalELBO(gpytorch.mlls.VariationalELBO):
    def __init__(self, likelihood, model, num_data, beta=1.0):
        super().__init__(likelihood, model, num_data)
        self.beta = beta

    def forward(self, variational_dist, target, *args, **kwargs):
        elbo = super().forward(variational_dist, target, *args, **kwargs)
        log_likelihood = variational_dist.log_prob(target).mean()
        kl_divergence = self.beta * torch.max(log_likelihood - elbo, torch.tensor(0.0).to(elbo.device))
        return  log_likelihood, kl_divergence
    
def train_model(model, likelihood, train_loader, val_loader, n_epochs, device, patience=100):
    # Freeze most of the feature extractor
    freeze_layers(model, num_layers_to_train=2)  # Adjust this number as needed
    print('Model', model)
    model.to(device)
    likelihood.to(device)
    
    optimizer = torch.optim.Adam([
        {'params': [param for param in model.feature_extractor.parameters() if param.requires_grad], 'lr': 1e-5, 'weight_decay': 1e-2},
        {'params': model.gp_layer.parameters(), 'lr': 1e-4},
        {'params': likelihood.parameters(), 'lr': 1e-4},
    ])
   
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    mll = CustomVariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset), beta=0.1)    

    best_val_loss = float('inf')
    early_stopping_counter = 0
   
    for epoch in range(1, n_epochs + 1):
        model.train()
        likelihood.train()
        train_loss, train_log_likelihood, train_kl_divergence = 0.0, 0.0, 0.0 
        
        for batch_idx, (data, target,_) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")):
            data, target = data.to(device), target.to(device).view(-1, 1)
            optimizer.zero_grad()
            
            # Log input statistics
            print(f"\nBatch {batch_idx}:")
            print(f"  Input - Mean: {data.mean().item():.4f}, Std: {data.std().item():.4f}")
            
            # Get features
            features = model.feature_extractor(data)
            print(f"  Features - Mean: {features.mean().item():.4f}, Std: {features.std().item():.4f}")
            
            # Get GP output
            output = model.prediction(data)
            print(f"  GP Mean - Mean: {output.mean.mean().item():.4f}, Std: {output.mean.std().item():.4f}")
            print(f"  GP Variance - Mean: {output.variance.mean().item():.4f}, Std: {output.variance.std().item():.4f}")
            
            # loss = -mll(output, target).mean()
            log_likelihood, kl_div = mll(output, target)
            # The ELBO is log_likelihood - kl_div
            elbo = log_likelihood - kl_div
            loss = -elbo.mean()            
        
            print(f"  Loss: {loss.item():.4f}")
            
            if not torch.isfinite(loss):
                print("Warning: non-finite loss, ending training ")
                return best_val_loss
            
            loss.backward()
            check_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Log gradients
            print("  Gradient Norms:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1e-4:  # Only print significant gradients
                        print(f"    {name}: {grad_norm:.4f}")
            
            optimizer.step()
            
            train_loss += loss.item()
            train_log_likelihood += log_likelihood.mean().item()
            train_kl_divergence += kl_div.mean().item()

        train_loss /= len(train_loader)
        train_log_likelihood /= len(train_loader)
        train_kl_divergence /= len(train_loader)
        
        val_loss, val_log_likelihood, val_kl_divergence = validate(model, likelihood, mll, val_loader, device)
        
        # print(f"\nEpoch {epoch} Summary:")
        # print(f"  Train Loss: {train_loss:.4f}")
        # print(f"  Val Loss: {val_loss:.4f}")
        
        # print("\nGP Hyperparameters:")
        print_gp_hyperparams(model)
        print_variational_params(model)
        
        print(f"Epoch {epoch}:")
        print(f"  Train - Loss: {train_loss:.4f}, Log-Likelihood: {train_log_likelihood:.4f}, KL Divergence: {train_kl_divergence:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Log-Likelihood: {val_log_likelihood:.4f}, KL Divergence: {val_kl_divergence:.4f}")

        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save({
                'model': model.state_dict(),
                'likelihood': likelihood.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'best_model.pth')
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            early_stopping_counter += 1
            print(f"EarlyStopping counter: {early_stopping_counter} out of {patience}")
        
        if epoch % 10 == 0:
            print_gp_hyperparams(model)
            for param_group in optimizer.param_groups:
                print(f"Learning rate for {param_group['params'][0].shape}: {param_group['lr']:.6f}")
        
        if early_stopping_counter >= patience:
            print("Early stopping")
            break
    
    print(f"Best validation loss: {best_val_loss:.4f}")
    return best_val_loss

def print_gp_hyperparams(model):
    for name, param in model.gp_layer.named_parameters():
        if 'raw' in name:
            if param.numel() == 1:
                print(f"{name}: {param.item():.4f} (transformed: {param.exp().item():.4f})")
            else:
                print(f"{name}:")
                print(f"  Raw - Mean: {param.mean().item():.4f}, Std: {param.std().item():.4f}")
                print(f"  Raw - Min: {param.min().item():.4f}, Max: {param.max().item():.4f}")
                print(f"  Transformed - Mean: {param.exp().mean().item():.4f}, Std: {param.exp().std().item():.4f}")
                print(f"  Transformed - Min: {param.exp().min().item():.4f}, Max: {param.exp().max().item():.4f}")

def validate(model, likelihood, mll, val_loader, device):
    model.eval()
    likelihood.eval()
    val_loss, val_log_likelihood, val_kl_divergence = 0.0, 0.0, 0.0 
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for data, target, ids in val_loader:
            data, target = data.to(device), target.to(device).view(-1, 1)
            output = model.prediction(data)
            log_likelihood, kl_div = mll(output, target)
            elbo = log_likelihood - kl_div
            loss = -elbo.mean()
            
            val_loss += loss.item()
            val_log_likelihood += log_likelihood.mean().item()
            val_kl_divergence += kl_div.mean().item()
    
    num_batches = len(val_loader)
    return (val_loss / num_batches, 
            val_log_likelihood / num_batches, 
            val_kl_divergence / num_batches)

def test_model(model, likelihood, test_loader, device):
    model.eval()
    likelihood.eval()
    
    results = defaultdict(lambda: defaultdict(list))
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for data, target, ids in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            
            # Make predictions
            output = likelihood(model(data))
            
            # Get mean and confidence region
            mean = output.mean
            std = output.stddev
            # Calculate 95% confidence interval
            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
            
            # Store results for each subject
            for i, subject_id in enumerate(ids):
                results[subject_id]['true'].append(target[i].item())
                results[subject_id]['mean'].append(mean[i].item())
                results[subject_id]['lower'].append(lower[i].item())
                results[subject_id]['upper'].append(upper[i].item())
    
    # Calculate overall metrics
    all_true = []
    all_pred = []
    for subject_id in results:
        all_true.extend(results[subject_id]['true'])
        all_pred.extend(results[subject_id]['mean'])
    
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    
    mse = mean_squared_error(all_true, all_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_true, all_pred)
    r2 = r2_score(all_true, all_pred)
    
    # Calculate coverage
    all_lower = [item for subj in results.values() for item in subj['lower']]
    all_upper = [item for subj in results.values() for item in subj['upper']]
    coverage = np.mean((all_true >= all_lower) & (all_true <= all_upper))
    
    print(f"Test Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"95% Confidence Interval Coverage: {coverage:.4f}")
    
    return results, mse, rmse, mae, r2, coverage

parser = argparse.ArgumentParser(description='Stochastic Variational Deep Kernel Regression for Neurodegeration Prediction') 
args = parser.parse_args()
parser.add_argument("--gpuid", help="GPUs", default=0)
parser.add_argument("--file", help="File to load the data", default='subjectsamples_longclean_hmuse_adniblsa')
parser.add_argument("--target", help="GPUs", default=13)
parser.add_argument("--task", help="GPUs", default='13')
parser.add_argument("--n_epochs", help="Number of epochs", default=100)
parser.add_argument("--lr_feature_extractor", type=float, default=0.001, help="Learning rate for the feature extractor")
parser.add_argument("--lr_gp", type=float, default=0.001, help="Learning rate for the GP components")
parser.add_argument("--personalization", help="Personalization", default=True)

expID = 'adniblsa'
args = parser.parse_args()
file = args.file
target = args.target
task = args.task 
gpu_id = args.gpuid
n_epochs = args.n_epochs

datasamples = pd.read_csv('/home/cbica/Desktop/LongGPClustering/data' + str(2) + '/' + file + '.csv')
longitudinal_covariates = pd.read_csv('/home/cbica/Desktop/LongGPClustering/data' + str(2) + '/longitudinal_covariates_subjectsamples_longclean_hmuse_convs_adniblsa.csv')
subject_ids = list(datasamples['PTID'].unique())

f = open('/home/cbica/Desktop/LongGPClustering/roi_to_idx.json')
roi_to_idx = json.load(f)
index_to_roi = {v: k for k, v in roi_to_idx.items()}
print(index_to_roi)

for fold in range(1):
    print('FOLD::', fold)
    train_ids, test_ids = [], []

    with open("/home/cbica/Desktop/LongGPClustering/data" + str(2) + "/train_subject_adniblsa_ids_hmuse" + "" + str(fold) + ".pkl", "rb") as openfile:
        while True:
            try:
                train_ids.append(pickle.load(openfile))
            except EOFError:
                break

    with open("/home/cbica/Desktop/LongGPClustering/data" + str(2) + "/test_subject_adniblsa_ids_hmuse" + "" + str(fold) + ".pkl", "rb") as openfile:
        while True:
            try:
                test_ids.append(pickle.load(openfile))
            except EOFError:
                break

    train_ids = train_ids[0]

    test_ids = test_ids[0]
    val_ids = train_ids[:100]
    train_ids = train_ids[100:]
    print('Train IDs', len(train_ids))
    print('Test IDs', len(test_ids))
    print('Validation IDs', len(val_ids))

    for t in test_ids:
        if t in train_ids:
            raise ValueError('Test Samples belong to the train!')

    train_x = datasamples[datasamples['PTID'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['PTID'].isin(train_ids)]['Y']
    test_x = datasamples[datasamples['PTID'].isin(test_ids)]['X']
    test_y = datasamples[datasamples['PTID'].isin(test_ids)]['Y']
    val_x = datasamples[datasamples['PTID'].isin(val_ids)]['X']
    val_y = datasamples[datasamples['PTID'].isin(val_ids)]['Y']

    corresponding_test_ids = datasamples[datasamples['PTID'].isin(test_ids)]['PTID'].to_list()
    corresponding_train_ids = datasamples[datasamples['PTID'].isin(train_ids)]['PTID'].to_list()
    corresponding_val_ids = datasamples[datasamples['PTID'].isin(val_ids)]['PTID'].to_list()

    train_x, train_y_all, test_x, test_y_all = process_temporal_singletask_data(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, test_ids=test_ids)
    val_x, val_y, val_x, val_y_all = process_temporal_singletask_data(train_x=val_x, train_y=val_y, test_x=val_x, test_y=val_y, test_ids=test_ids)

    if torch.cuda.is_available():
        train_x = train_x.cuda(gpu_id) 
        train_y_all = train_y_all.cuda(gpu_id)#.squeeze()
        test_x = test_x.cuda(gpu_id) 
        test_y_all = test_y_all.cuda(gpu_id)#.squeeze() 
        val_x = val_x.cuda(gpu_id)
        val_y_all = val_y_all.cuda(gpu_id)

    if task == 'SPARE_AD':
        roi_idx = 0
    elif task == 'SPARE_BA':
        roi_idx = 1
    else:
        print('To infer a muse roi')
        roi_idx = int(task) 
        print('ROI:', 'H_MUSE_Volume_' +  str(index_to_roi[int(roi_idx)])) 

    target = task

    test_y = test_y_all[:, roi_idx]
    train_y = train_y_all[:, roi_idx]
    val_y = val_y[:, roi_idx]


    # store the data
    # print('Train X:', train_x[:5,:])

    # print('Train Y:', train_y[:5])


    # Initialize your custom datasets
    train_dataset = TrainDataset(train_x, train_y, corresponding_train_ids)
    test_dataset = TestDataset(test_x, test_y, corresponding_test_ids)
    val_dataset = TestDataset(val_x, val_y, corresponding_val_ids)

    # Create DataLoaders for your datasets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    #### Model Definition ####
    # Usage
    input_dim = train_x.shape[1]  # Your input dimension
    gp_hidden_dim = 64  
    inducing_points = int(0.3 * train_x.shape[0])  # Number of inducing points
    feature_extractor = MLP_Reduced(input_dim, gp_hidden_dim)
    model = DKLModel(feature_extractor, gp_hidden_dim, inducing_points)

    # Define the likelihood for regression
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    # If you run this example without CUDA, I hope you like waiting!
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    n_epochs = 500
    ## Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    ## Train the model ##
    best_val_loss = train_model(model, likelihood, train_loader, val_loader, n_epochs=1000, device=device, patience=30)
        
    ## Test the best model ##
    model.eval
    likelihood.eval

    # Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model'])
    likelihood.load_state_dict(checkpoint['likelihood'])

    model.to(device)
    likelihood.to(device)

    results, mse, rmse, mae, r2, coverage = test_model(model, likelihood, test_loader, device)

    # Save results
    import json

    with open('test_results.json', 'w') as f:
        json.dump(results, f)

    print(f"Results saved to test_results.json")


