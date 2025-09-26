### SVDKRegression (o1 version)
# svdk_regression.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
import pandas as pd
import gpytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from functions import process_temporal_singletask_data  # Assuming you have this function defined in a separate file
import json 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
import pandas as pd
import pickle
import argparse
from sklearn.preprocessing import StandardScaler

## Set double precision 
torch.set_default_dtype(torch.float64)

# Step 1: Define the CognitiveDataset class
class CognitiveDataset(Dataset):
    def __init__(self, inputs, targets, subject_ids):
        """
        inputs: NumPy array of input features, shape (num_samples, input_dim)
        targets: NumPy array of target values, shape (num_samples,)
        subject_ids: List or array of subject IDs, length num_samples
        """
        assert len(inputs) == len(targets) == len(subject_ids), "Inputs, targets, and subject_ids must have the same length."
        
        self.inputs = torch.tensor(inputs, dtype=torch.float64)
        self.targets = torch.tensor(targets, dtype=torch.float64)
        self.subject_ids = subject_ids  # List or array of subject IDs
        
        # Create a mapping from subject ID to indices
        self.subject_to_indices = {}
        for idx, subject_id in enumerate(self.subject_ids):
            if subject_id not in self.subject_to_indices:
                self.subject_to_indices[subject_id] = []
            self.subject_to_indices[subject_id].append(idx)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):

        return self.inputs[idx], self.targets[idx], self.subject_ids[idx]

# Step 2: Define the FeatureExtractor and RegressionNN classes
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # print(f"FeatureExtractor input shape: {x.shape}")
        # print(f"Input dtype: {x.dtype}")
        # print(f"Model parameters dtype: {self.fc1.weight.dtype}")
        return self.relu(self.fc1(x))

class RegressionNN(nn.Module):
    def __init__(self, feature_extractor):
        super(RegressionNN, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc2 = nn.Linear(feature_extractor.fc1.out_features, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.fc2(features).squeeze(-1)

# Step 3: Define the SubjectBatchSampler
class SubjectBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subject_ids = list(dataset.subject_to_indices.keys())
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.subject_ids)
        batch = []
        for subject_id in self.subject_ids:
            indices = self.dataset.subject_to_indices[subject_id]
            batch.extend(indices)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# Step 4: Define the DeepKernelGPModel
class DeepKernelGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, feature_extractor):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(DeepKernelGPModel, self).__init__(variational_strategy)
        
        self.feature_extractor = feature_extractor  # Pretrained feature extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    def forward(self, x):
        # print('Input shape to the GP:', x.shape)
        projected_x = self.feature_extractor(x)
    
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Step 4: Define the DeepKernel class
class DeepKernel(gpytorch.kernels.Kernel):
    def __init__(self, base_kernel, feature_extractor):
        super(DeepKernel, self).__init__()
        self.base_kernel = base_kernel
        self.feature_extractor = feature_extractor

    def forward(self, x1, x2, **params):
        x1_ = self.feature_extractor(x1)
        x2_ = self.feature_extractor(x2)
        return self.base_kernel(x1_, x2_)

# Step 5: Define the GPModelWrapper
class GPModelWrapper(nn.Module):
    def __init__(self, gp_model, likelihood):
        super(GPModelWrapper, self).__init__()
        self.gp_model = gp_model
        self.likelihood = likelihood
    
    def forward(self, x):
        # print('GP Model Input Shape:', x.shape)
        return self.gp_model(x)

def load_and_preprocess_data(folder, file, train_ids, test_ids, single_muse):
    
    f = open('/home/cbica/Desktop/LongGPClustering/roi_to_idx.json')
    roi_to_idx = json.load(f)

    print(roi_to_idx)

    index_to_roi = {v: k for k, v in roi_to_idx.items()}

    print(index_to_roi)

    # Load your data
    datasamples = pd.read_csv('/home/cbica/Desktop/LongGPClustering/data' + str(folder) + '/' + file + '.csv')
    
    # Set up the train/test data
    train_x = datasamples[datasamples['PTID'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['PTID'].isin(train_ids)]['Y']
    test_x = datasamples[datasamples['PTID'].isin(test_ids)]['X']
    test_y = datasamples[datasamples['PTID'].isin(test_ids)]['Y']
    
    # Corresponding subject IDs
    corresponding_train_ids = datasamples[datasamples['PTID'].isin(train_ids)]['PTID'].tolist()
    corresponding_test_ids = datasamples[datasamples['PTID'].isin(test_ids)]['PTID'].tolist()
    
    # Process the data
    train_x, train_y_all, test_x, test_y_all = process_temporal_singletask_data(
        train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, test_ids=test_ids
    )
    
    # Convert tensors to numpy arrays
    train_x = train_x.numpy()
    train_y_all = train_y_all.numpy()
    test_x = test_x.numpy()
    test_y_all = test_y_all.numpy()
    
    # Select the specific ROI
    single_muse = 'H_MUSE_Volume_47'
    print('ROI NAME', single_muse)
    
    if single_muse == 'SPARE_AD':
        list_index = 0
    elif single_muse == 'SPARE_BA':
        list_index = 1
    else:
        print('MUSE ROI', single_muse)
        list_index = roi_to_idx[single_muse.split('_')[-1]]
    
    train_y = train_y_all[:, list_index]
    test_y = test_y_all[:, list_index]
    
    return train_x, train_y, test_x, test_y, corresponding_train_ids, corresponding_test_ids

def select_inducing_points(train_x, train_subject_ids, selected_subject_ids=None, num_points_per_subject=3):
    """
    Selects inducing points by sampling observations from selected subjects.

    Parameters:
    - train_x: NumPy array of shape (num_samples, input_dim)
    - train_subject_ids: List or array of subject IDs corresponding to each row in train_x
    - selected_subject_ids: List of subject IDs from which to select inducing points (optional)
    - num_points_per_subject: Number of inducing points to select per subject

    Returns:
    - inducing_points: Torch tensor of selected inducing points
    """
    import pandas as pd
    import numpy as np
    import torch

    print('Selected Subjects:', len(selected_subject_ids))

    # Determine the expected number of features
    expected_num_features = train_x.shape[1]

    # Create a DataFrame for easier manipulation
    data = pd.DataFrame(train_x)
    data['PTID'] = train_subject_ids

    inducing_points_list = []

    if selected_subject_ids is not None:
        # Filter data to include only selected subjects
        data_selected = data[data['PTID'].isin(selected_subject_ids)]
    else:
        data_selected = data

    # Group by subject
    grouped = data_selected.groupby('PTID')

    for subject_id, group in grouped:
        # Sort the group by temporal variable (assuming it's the last column minus one)
        temporal_col_index = expected_num_features - 2  # Adjust index if necessary
        group_sorted = group.sort_values(by=group.columns[temporal_col_index])
        num_observations = group_sorted.shape[0]

        if num_observations >= num_points_per_subject:
            # Select earliest, median, and latest time points
            indices = [0, num_observations // 2, num_observations - 1]
            selected_points = group_sorted.iloc[indices]
        else:
            # If less than num_points_per_subject observations, select all
            selected_points = group_sorted

        # Drop 'SubjectID' column and convert to values
        selected_values = selected_points.drop('PTID', axis=1).values
    
        if selected_values.size > 0:
            if selected_values.shape[1] != expected_num_features:
                print(f"Warning: Subject {subject_id} has unexpected number of features: {selected_values.shape[1]} (expected {expected_num_features}).")
                continue  # Skip this subject or handle as appropriate

            # if np.isnan(selected_values).any() or np.isinf(selected_values).any():
            #     print(f"Warning: NaNs or Infs detected in data for subject {subject_id}.")
            #     continue  # Skip or handle as appropriate

            # Ensure numerical data type
            selected_values = selected_values.astype(np.float32)
            inducing_points_list.append(selected_values)
        else:
            print(f"Warning: No inducing points for subject {subject_id}.")

    if inducing_points_list:
        inducing_points_array = np.vstack(inducing_points_list)
        inducing_points = torch.tensor(inducing_points_array, dtype=torch.float32)
    else:
        raise ValueError("No inducing points were selected. Check your data and selection criteria.")

    return inducing_points


class DeepKernelGPModelSeparateKernels(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, feature_extractor):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(DeepKernelGPModelSeparateKernels, self).__init__(variational_strategy)
        
        self.feature_extractor = feature_extractor  # Pretrained feature extractor
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Separate kernels for baseline and temporal features
        self.covar_module_baseline = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.covar_module_time = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    
    def forward(self, x):
        # Split baseline and temporal features
        x_baseline = x[:, :-1]  # Assuming temporal variable is the last column
        x_time = x[:, -1].unsqueeze(1)
        
        print(f"Input shape: {x.shape}, Baseline shape: {x_baseline.shape}, Time shape: {x_time.shape}")

        # Apply feature extractor to baseline features
        projected_x_baseline = self.feature_extractor(x_baseline)
        
        # Compute mean
        mean_x = self.mean_module(projected_x_baseline)
        
        # Compute covariance matrices
        covar_baseline = self.covar_module_baseline(projected_x_baseline)
        covar_time = self.covar_module_time(x_time)
        
        # Combine covariance matrices (choose either multiplication or addition)
        covar_x = covar_baseline * covar_time  # Multiplicative kernel
        # covar_x = covar_baseline + covar_time  # Additive kernel
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Step 6: Main function
def main():

    fold = 0 
    # wandb.init(project="HMUSEDeepSingleTask", entity="vtassop", save_code=True)
    parser = argparse.ArgumentParser(description='Deep Regression for Neurodegeneration Prediction')
    ## Data Parameters 
    parser.add_argument("--experimentID", help="Indicates the Experiment Identifier", default='adniblsa') # 1adni normally
    parser.add_argument("--file", help="Identifier for the data", default="subjectsamples_longclean_hmuse_adniblsa")
    parser.add_argument("--folder", type=int, default=2)

    args = parser.parse_args()
    expID = args.experimentID
    file = args.file
    folder = args.folder
    train_ids, test_ids = [], []
    
    ## Store the trajectories 
    population_results = {'id': [], 'score': [], 'fold': [], 'y': [], 'time': [], 'variance': []} 
    population_mae_kfold = {''}
    
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

    print('Train IDs:', len(train_ids))
    print('Test IDs:', len(test_ids))

    train_ids = train_ids[0]
    test_ids = test_ids[0]

    train_x, train_y, test_x, test_y, corresponding_train_ids, corresponding_test_ids = load_and_preprocess_data(folder=folder, file=file, train_ids=train_ids, test_ids=test_ids, single_muse='H_MUSE_Volume_47')

    # Assuming temporal variable is the last column
    temporal_index = -1

    # Split baseline and temporal features
    train_x_baseline = train_x[:, :-1]
    train_x_time = train_x[:, temporal_index].reshape(-1, 1)
    test_x_baseline = test_x[:, :-1]
    test_x_time = test_x[:, temporal_index].reshape(-1, 1)

    # Standardize temporal variable
    scaler_time = StandardScaler()
    train_x_time = scaler_time.fit_transform(train_x_time)
    test_x_time = scaler_time.transform(test_x_time)

    # Combine features back
    train_x = np.hstack((train_x_baseline, train_x_time))
    test_x = np.hstack((test_x_baseline, test_x_time))

    # Convert data tensors
    train_x = torch.tensor(train_x, dtype=torch.float64)
    train_y = torch.tensor(train_y, dtype=torch.float64)
    test_x = torch.tensor(test_x, dtype=torch.float64)
    test_y = torch.tensor(test_y, dtype=torch.float64)
        
    train_x = train_x.double()
    train_y = train_y.double()
    test_x = test_x.double()
    test_y = test_y.double()

    if np.isnan(train_x).any() or np.isinf(train_x).any():
        print("Warning: NaNs or Infs detected in train_x.")
    if np.isnan(test_x).any() or np.isinf(test_x).any():
        print("Warning: NaNs or Infs detected in test_x.")

    # Assuming train_x, train_y, and corresponding_train_ids are defined and aligned
    train_dataset = CognitiveDataset(inputs=train_x, targets=train_y, subject_ids=corresponding_train_ids)

    # Similarly for the test dataset
    test_dataset = CognitiveDataset(inputs=test_x, targets=test_y, subject_ids=corresponding_test_ids)

    batch_size = 64  # Adjust as needed
    train_sampler = SubjectBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine input dimension
    input_dim = train_x.shape[1]
    hidden_dim = 128  # Adjust as needed

    # =======================================
    # Step 1: Train the Deep Regression Model
    # =======================================
    # Initialize the model
    feature_extractor = FeatureExtractor(input_dim, hidden_dim)
    model = RegressionNN(feature_extractor)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop for deep regression model
    num_epochs = 10  # Adjust as needed
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Regression Loss: {epoch_loss:.4f}")

    # Save the feature extractor
    torch.save(feature_extractor.state_dict(), 'feature_extractor.pth')

    # =======================================
    # Step 2: Load Feature Extractor for GP Model
    # =======================================
    # Re-initialize the feature extractor and load the saved parameters
    feature_extractor_gp = FeatureExtractor(input_dim, hidden_dim)
    feature_extractor_gp.load_state_dict(torch.load('feature_extractor.pth'))
    feature_extractor_gp.eval()

    # Optionally, freeze the feature extractor
    # for param in feature_extractor_gp.parameters():
    #     param.requires_grad = False

    # Prepare the inducing points
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)

    unique_train_subject_ids = list(set(corresponding_train_ids))
    selected_subject_ids = random.sample(unique_train_subject_ids, 200)

    inducing_points = select_inducing_points(train_x, corresponding_train_ids,selected_subject_ids=selected_subject_ids, num_points_per_subject=3)

    inducing_points_np = inducing_points.numpy()
    # Convert inducing points
    inducing_points = inducing_points.double()

    unique_inducing_points = np.unique(inducing_points_np, axis=0)
    if unique_inducing_points.shape[0] < inducing_points_np.shape[0]:
        print("Warning: Duplicated inducing points detected.")

    # Define the base kernel
    base_kernel = gpytorch.kernels.RBFKernel()

    # Define the deep kernel
    deep_kernel = DeepKernel(base_kernel, feature_extractor_gp)

    # Define the inducing point kernel
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = gpytorch.kernels.InducingPointKernel(
        deep_kernel,
        inducing_points=inducing_points,
        likelihood=likelihood
    )

    # =======================================
    # Step 3: Initialize the GP Model and Likelihood
    # =======================================
    gp_model = DeepKernelGPModel(inducing_points, feature_extractor_gp)

    # Define the model wrapper and optimizer
    model_wrapper = GPModelWrapper(gp_model, likelihood)

    # Convert model components to double precision
    feature_extractor = feature_extractor.double()
    gp_model = gp_model.double()
    likelihood = likelihood.double()

    # Ensure model wrapper is in double precision
    model_wrapper = model_wrapper.double()

    mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=len(train_dataset))
    optimizer = torch.optim.Adam([
        {'params': gp_model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=1e-3)

    inducing_points_np = inducing_points.numpy()
    rank = np.linalg.matrix_rank(inducing_points_np)
    print(f"Rank of inducing points matrix: {rank}")

    torch.set_default_dtype(torch.float64)
    inducing_points = torch.tensor(inducing_points_np, dtype=torch.float64)
    # =======================================
    # Step 4: Training Loop for GP Model
    # =======================================
    torch.nn.utils.clip_grad_norm_(gp_model.parameters(), max_norm=1.0)

    num_epochs = 100  # Adjust as needed
    for epoch in range(num_epochs):
        model_wrapper.train()
        likelihood.train()
        running_loss = 0.0
        for inputs, targets, _ in train_loader:
            # print(f"GP Batch input shape: {inputs.shape}")
            # print(f"GP Batch target shape: {targets.shape}")
            optimizer.zero_grad()
            output = model_wrapper(inputs)
            loss = -mll(output, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            with torch.no_grad():
                K = gp_model.covar_module(feature_extractor_gp(inducing_points)).evaluate()
                eigenvalues = torch.linalg.eigvalsh(K)
                # print("Eigenvalues of the covariance matrix:", eigenvalues)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, GP Loss: {epoch_loss:.4f}")

    # =======================================
    # Step 5: Evaluation
    # =======================================
    model_wrapper.eval()
    likelihood.eval()
    with torch.no_grad():
        predictions = []
        actuals = []
        for inputs, targets, _ in test_loader:
            output = model_wrapper(inputs)
            pred = likelihood(output)
            predictions.extend(pred.mean.numpy())
            actuals.extend(targets.numpy())

            ### Store the predictions and groundtruth for each subject 




    # Calculate evaluation metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)

    ## Store the Evaluation Metrics across the whole fold, for 5-Fold Visualization
    


    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Optionally, save the GP model
    torch.save(gp_model.state_dict(), 'svdk_gp_model.pth')
    torch.save(likelihood.state_dict(), 'svdk_likelihood.pth')

if __name__ == "__main__":
    main()
