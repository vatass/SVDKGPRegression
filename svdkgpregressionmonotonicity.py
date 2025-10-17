# svdk_regression_monotonic_with_loss_print.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
import pandas as pd
import gpytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from functions import process_temporal_singletask_data  # Assuming you have this function defined
import json
import pickle
import argparse
from sklearn.preprocessing import StandardScaler

# Set double precision
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
        return self.relu(self.fc1(x))
class RegressionNN(nn.Module):
    def __init__(self, feature_extractor):
        super(RegressionNN, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc2 = nn.Linear(feature_extractor.fc1.out_features, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.fc2(features).squeeze(-1)


# Feature Extractor: Latent Concatenation of Time and Imaging Features
class FeatureExtractorLatentConcatenation(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractorLatentConcatenation, self).__init__()
        # Assuming the last feature is time
        self.imaging_dim = input_dim - 1
        self.time_dim = 1
        self.hidden_dim = hidden_dim

        # Layers for imaging features
        self.imaging_fc = nn.Sequential(
            nn.Linear(self.imaging_dim, hidden_dim),
            nn.ReLU(),
            # You can add more layers if needed
        )

        # Layers for time feature
        self.time_fc = nn.Sequential(
            nn.Linear(self.time_dim, hidden_dim),
            nn.ReLU(),
            # You can add more layers if needed
        )

        # Final layers after concatenation
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            # You can add more layers if needed
        )

    def forward(self, x):
        x_imaging = x[:, :-1]  # All features except the last one
        x_time = x[:, -1].unsqueeze(1)  # The last feature (time)

        # Process imaging features
        imaging_features = self.imaging_fc(x_imaging)

        # Process time feature
        time_features = self.time_fc(x_time)

        # Concatenate imaging and time features
        combined_features = torch.cat([imaging_features, time_features], dim=1)

        # Final processing
        output = self.fc_combined(combined_features)
        return output

class RegressionNNLatentConcatenation(nn.Module):
    def __init__(self, feature_extractor):
        super(RegressionNNLatentConcatenation, self).__init__()
        self.feature_extractor = feature_extractor

        # The output dimension of feature_extractor is hidden_dim
        self.fc_out = nn.Linear(feature_extractor.hidden_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.fc_out(features).squeeze(-1)
        return output

# Separate Processing and Addition
class FeatureExtractorLatentAddition(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractorLatentAddition, self).__init__()
        self.imaging_dim = input_dim - 1
        self.time_dim = 1

        # Imaging features layers
        self.imaging_fc = nn.Sequential(
            nn.Linear(self.imaging_dim, hidden_dim),
            nn.ReLU(),
            # Additional layers if needed
        )

        # Time feature layers
        self.time_fc = nn.Sequential(
            nn.Linear(self.time_dim, hidden_dim),
            nn.ReLU(),
            # Additional layers if needed
        )

        # Final activation (optional)
        self.final_activation = nn.ReLU()

    def forward(self, x):
        x_imaging = x[:, :-1]  # Imaging features
        x_time = x[:, -1].unsqueeze(1)  # Time feature

        # Process features separately
        imaging_features = self.imaging_fc(x_imaging)
        time_features = self.time_fc(x_time)

        # Combine features by addition
        combined_features = imaging_features + time_features
        output = self.final_activation(combined_features)
        return output

class RegressionNNLatentAddition(nn.Module):
    def __init__(self, feature_extractor):
        super(RegressionNNLatentAddition, self).__init__()
        self.feature_extractor = feature_extractor

        # The output dimension of feature_extractor is hidden_dim
        self.fc_out = nn.Linear(feature_extractor.hidden_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.fc_out(features).squeeze(-1)
        return output

# Interaction 
class FeatureExtractorInteraction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractorInteraction, self).__init__()
        self.imaging_dim = input_dim - 1
        self.time_dim = 1

        # Imaging features layers
        self.imaging_fc = nn.Sequential(
            nn.Linear(self.imaging_dim, hidden_dim),
            nn.ReLU(),
            # Additional layers if needed
        )

        # Time feature layers
        self.time_fc = nn.Sequential(
            nn.Linear(self.time_dim, hidden_dim),
            nn.ReLU(),
            # Additional layers if needed
        )

        # Interaction layer
        self.fc_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Additional layers if needed
        )

    def forward(self, x):
        x_imaging = x[:, :-1]  # Imaging features
        x_time = x[:, -1].unsqueeze(1)  # Time feature

        # Process features separately
        imaging_features = self.imaging_fc(x_imaging)
        time_features = self.time_fc(x_time)

        # Element-wise multiplication (interaction)
        interaction = imaging_features * time_features

        # Process interaction
        output = self.fc_interaction(interaction)
        return output

class RegressionNNInteraction(nn.Module):
    def __init__(self, feature_extractor):
        super(RegressionNNInteraction, self).__init__()
        self.feature_extractor = feature_extractor

        # The output dimension of feature_extractor is hidden_dim
        self.fc_out = nn.Linear(feature_extractor.hidden_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.fc_out(features).squeeze(-1)
        return output

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

# Step 2: Define the SubjectBatchSampler
class TestSubjectBatchSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.subject_ids = list(dataset.subject_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.subject_ids)
        for subject_id in self.subject_ids:
            indices = self.dataset.subject_to_indices[subject_id]
            yield indices  # Yield all indices for the current subject

    def __len__(self):
        return len(self.subject_ids)


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
        return self.gp_model(x)


### Util functions ###
# Function to calculate rate of change for the original SPARE-AD values
def calculate_rate_of_change(age, biomarker):
    slope, intercept = np.polyfit(age,biomarker, 1)
    return slope

def load_and_preprocess_data(folder, file, train_ids, test_ids, single_muse):
    f = open('/home/cbica/Desktop/LongGPClustering/roi_to_idx.json')
    roi_to_idx = json.load(f)

    index_to_roi = {v: k for k, v in roi_to_idx.items()}

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

    if single_muse == 'SPARE_AD':
        list_index = 0
    elif single_muse == 'SPARE_BA':
        list_index = 1
    else:
        list_index = roi_to_idx[single_muse.split('_')[-1]]

    train_y = train_y_all[:, list_index]
    test_y = test_y_all[:, list_index]

    return train_x, train_y, test_x, test_y, corresponding_train_ids, corresponding_test_ids

def calculate_coverage(targets, lower, upper):
    """
    Calculate the coverage of the predictive confidence intervals.

    Args:
        targets (np.ndarray): True target values.
        lower (np.ndarray): Lower bounds of the confidence intervals.
        upper (np.ndarray): Upper bounds of the confidence intervals.

    Returns:
        coverage (float): Coverage rate as a percentage.
    """
    within_interval = (targets >= lower) & (targets <= upper)
    coverage = np.mean(within_interval) * 100
    return coverage

def bootstrap_metrics(predictions, targets, n_bootstrap=1000, alpha=0.05):
    """
    Compute bootstrap confidence intervals for RMSE, MAE, and R².

    Args:
        predictions (np.ndarray): Model predictions.
        targets (np.ndarray): True target values.
        n_bootstrap (int): Number of bootstrap samples.
        alpha (float): Significance level for confidence intervals.

    Returns:
        metrics_ci (dict): Confidence intervals for each metric.
    """
    n_samples = len(targets)
    rmse_samples = []
    mae_samples = []
    r2_samples = []

    # convert liusts to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets.detach().cpu())


    print("Type of predictions:", type(predictions))
    print("Shape of predictions:", predictions.shape)
    print("Type of targets:", type(targets))
    print("Shape of targets:", targets.shape)

    for _ in range(n_bootstrap):
        # Generate bootstrap sample indices
        indices = np.random.randint(0, n_samples, n_samples)
        pred_sample = predictions[indices]
        target_sample = targets[indices]

        # Compute metrics
        rmse = np.sqrt(np.mean((pred_sample - target_sample) ** 2))
        mae = np.mean(np.abs(pred_sample - target_sample))
        ss_res = np.sum((target_sample - pred_sample) ** 2)
        ss_tot = np.sum((target_sample - np.mean(target_sample)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # Store metrics
        rmse_samples.append(rmse)
        mae_samples.append(mae)
        r2_samples.append(r2)

    # Compute confidence intervals
    lower = alpha / 2 * 100
    upper = (1 - alpha / 2) * 100

    rmse_ci = np.percentile(rmse_samples, [lower, upper])
    mae_ci = np.percentile(mae_samples, [lower, upper])
    r2_ci = np.percentile(r2_samples, [lower, upper])

    metrics_ci = {
        'rmse': rmse_ci,
        'mae': mae_ci,
        'r2': r2_ci
    }

    return metrics_ci

def select_inducing_points(train_x, train_subject_ids, selected_subject_ids=None, num_points_per_subject=3):
    import pandas as pd
    import numpy as np
    import torch

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

        # Drop 'PTID' column and convert to values
        selected_values = selected_points.drop('PTID', axis=1).values

        if selected_values.size > 0:
            if selected_values.shape[1] != expected_num_features:
                print(f"Warning: Subject {subject_id} has unexpected number of features: {selected_values.shape[1]} (expected {expected_num_features}).")
                continue  # Skip this subject or handle as appropriate

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

def collate_fn(batch):
    # 'batch' is a list of samples, where each sample is a tuple (input, target, subject_id)
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    subject_ids = [item[2] for item in batch]  # Keep as list or convert to tensor if numeric
    
    return inputs, targets, subject_ids


# Step 6: Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fold = 0
    parser = argparse.ArgumentParser(description='Deep Regression for Neurodegeneration Prediction')
    # Data Parameters
    parser.add_argument("--experimentID", help="Indicates the Experiment Identifier", default='allstudies')
    parser.add_argument("--file", help="Identifier for the data", default="subjectsamples_longclean_hmuse_allstudies")
    parser.add_argument("--folder", type=int, default=1)
    parser.add_argument("--lambda_penalty", type=float, default=1)

    args = parser.parse_args()
    expID = args.experimentID
    file = args.file
    folder = args.folder
    lambda_penalty = args.lambda_penalty


    longitudinal_covariates = pd.read_csv('/home/cbica/Desktop/LongGPRegressionBaseline/longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv')
    longitudinal_covariates['Diagnosis'].replace([-1.0, 0.0, 1.0, 2.0], ['UKN', 'CN', 'MCI', 'AD'], inplace=True)

    population_results = {'id': [],'fold': [], 'score': [], 'y': [], 'variance': [], 'time': [], 'age': [] }
    population_fold_metrics = {'fold': [] , 'mse': [], 'mae': [], 'r2': [], 'coverage': [], 'interval': [], 'mean_roc_dev': []}
    population_per_subject_metrics = {'id': [], 'fold':[], 'mae': [], 'mse': [], 'coverage': [], 'interval': [], 'roc_dev': [], 'gt_roc': [], 'pred_roc': [], 'monotonicity': [], "timesteps": [], "sequence": [], "true_sequence": [], "subject_time": []}

    train_ids, test_ids = [], []
    # Load train IDs
    # with (open("./data"+str(folder)+"/train_subject_ids_hmuse_" + kfoldID + str(fold) +  ".pkl", "rb")) as openfile:
    with (open("/home/cbica/Desktop/LongGPClustering/data"+str(folder)+"/train_subject_allstudies_ids_hmuse" + str(fold) +  ".pkl", "rb")) as openfile:
        while True:
            try:
                train_ids.append(pickle.load(openfile))
            except EOFError:
                break 
      
    # with (open("./data"+str(folder)+"/test_subject_ids_hmuse_" + kfoldID + str(fold) + ".pkl", "rb")) as openfile:
    with (open("/home/cbica/Desktop/LongGPClustering/data"+str(folder)+"/test_subject_allstudies_ids_hmuse" + str(fold) +  ".pkl", "rb")) as openfile:

        while True:
            try:
                test_ids.append(pickle.load(openfile))
            except EOFError:
                break

    train_ids = train_ids[0]
    test_ids = test_ids[0]

    train_x, train_y, test_x, test_y, corresponding_train_ids, corresponding_test_ids = load_and_preprocess_data(
        folder=folder, file=file, train_ids=train_ids, test_ids=test_ids, single_muse='H_MUSE_Volume_47'
    )

    # Assuming temporal variable is the last column
    temporal_index = -1

    # Split baseline and temporal features
    train_x_baseline = train_x[:, :-1]
    train_x_time = train_x[:, temporal_index].reshape(-1, 1)
    test_x_baseline = test_x[:, :-1]
    test_x_time = test_x[:, temporal_index].reshape(-1, 1)

    # Standardize temporal variable
    # scaler_time = StandardScaler()
    # train_x_time = scaler_time.fit_transform(train_x_time)
    # test_x_time = scaler_time.transform(test_x_time)

    # Combine features back
    train_x = np.hstack((train_x_baseline, train_x_time))
    test_x = np.hstack((test_x_baseline, test_x_time))

    # Convert data tensors
    train_x = torch.tensor(train_x, dtype=torch.float64)
    train_y = torch.tensor(train_y, dtype=torch.float64)
    test_x = torch.tensor(test_x, dtype=torch.float64)
    test_y = torch.tensor(test_y, dtype=torch.float64)

    # Ensure double precision
    train_x = train_x.double()
    train_y = train_y.double()
    test_x = test_x.double()
    test_y = test_y.double()

    # Create datasets
    train_dataset = CognitiveDataset(inputs=train_x, targets=train_y, subject_ids=corresponding_train_ids)
    test_dataset = CognitiveDataset(inputs=test_x, targets=test_y, subject_ids=corresponding_test_ids)

    batch_size = 64  # Adjust as needed
    train_sampler = SubjectBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    test_subject_sampler = TestSubjectBatchSampler(test_dataset, shuffle=False)

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=pin, num_workers=2)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_subject_sampler,
        collate_fn=collate_fn, pin_memory=pin, num_workers=2
)
    # Determine input dimension
    input_dim = train_x.shape[1]
    hidden_dim = 128  # Adjust as needed

    # =======================================
    # Step 1: Train the Deep Regression Model
    # =======================================
    # Initialize the model
    feature_extractor = FeatureExtractorLatentConcatenation(input_dim, hidden_dim)
    model = RegressionNNLatentConcatenation(feature_extractor).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop for deep regression model
    num_epochs = 30  # Adjust as needed
    total_regression_loss = [] 
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets, _ in train_loader:

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        total_regression_loss.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Regression Loss: {epoch_loss:.4f}")

    # Save the feature extractor
    torch.save(feature_extractor.state_dict(), 'feature_extractor_latentconcatenation.pth')

    # Visualize the training loss
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(total_regression_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()
    plt.savefig('regression_training_loss.png',  dpi=300)
    plt.savefig('regression_training_loss.svg',  dpi=300)

    # =======================================
    # Step 2: Load Feature Extractor for GP Model
    # =======================================
    # Re-initialize the feature extractor and load the saved parameters
    feature_extractor_gp = FeatureExtractorLatentConcatenation(input_dim, hidden_dim).to(device)
    feature_extractor_gp.load_state_dict(torch.load('feature_extractor_latentconcatenation.pth'))
    feature_extractor_gp.eval()

    # Prepare the inducing points
    unique_train_subject_ids = list(set(corresponding_train_ids))
    selected_subject_ids = random.sample(unique_train_subject_ids, 200)

    inducing_points = select_inducing_points(
        train_x.numpy(), corresponding_train_ids, selected_subject_ids=selected_subject_ids, num_points_per_subject=3
    )

    # Convert inducing points
    inducing_points = inducing_points.double()

    # Define the likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    # =======================================
    # Step 3: Initialize the GP Model and Likelihood
    # =======================================
    gp_model = DeepKernelGPModel(inducing_points, feature_extractor_gp).to(device)

    # Define the model wrapper and optimizer
    model_wrapper = GPModelWrapper(gp_model, likelihood).to(device)

    # Ensure model components are in double precision
    feature_extractor = feature_extractor.double()
    gp_model = gp_model.double()
    likelihood = likelihood.double()
    model_wrapper = model_wrapper.double()

    mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=len(train_dataset))

    torch.set_default_dtype(torch.float64)
    

    # =======================================
    # Step 4: Training Loop for GP Model with Monotonicity Constraint
    # =======================================
    # Adjusted Training Loop
    num_epochs = 200
    learning_rate = 1e-4  # Reduced learning rate
    m = torch.nn.Softplus() #Replaced ReLU with Softplus to combat 0 gradients
    optimizer = torch.optim.Adam([
        {'params': gp_model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate)

    epoch_data_losses = []
    epoch_penalty_losses = []
    total_losses = []

    df_dt_means = []
    df_dt_stds = []
    df_dt_values_over_epochs = []  # To store df/dt values at selected epochs
    epochs_to_record = [1, 50, 100, 150, 200]  # Epochs at which to store df/dt values for histograms

    for epoch in range(num_epochs):
        model_wrapper.train()
        likelihood.train()
        running_loss = 0.0
        running_data_loss = 0.0
        running_penalty_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True).clone().detach().requires_grad_(True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with gpytorch.settings.fast_pred_var(False):
                output = model_wrapper(inputs)
                loss = -mll(output, targets)

                mean_output = output.mean

                df_dx = torch.autograd.grad(
                    outputs=mean_output,
                    inputs=inputs,
                    grad_outputs=torch.ones_like(mean_output),
                    create_graph=True
                )[0]
                # print(df_dx.shape)
                # print('Time', inputs[:, -1])

                df_dt = df_dx[:, -1]
                # print(f"df_dt values: {df_dt}")
                # check if df_dt contains NaN or Inf values
                # if torch.isnan(df_dt).any() or torch.isinf(df_dt).any():
                #     print(f"NaN or Inf values detected in df_dt. Skipping this iteration.")
                #     continue
                # import matplotlib.pyplot as plt

                # plt.hist(df_dt.detach().cpu().numpy(), bins=50)
                # plt.title('Histogram of df_dt')
                # plt.xlabel('df_dt')
                # plt.ylabel('Frequency')
                # plt.savefig('df_dt_histogram_epoch_'+str(epoch)+'.png')

                # Statistics of the Gradient !
                mean_df_dt = df_dt.mean().item()
                min_df_dt = df_dt.min().item()
                max_df_dt = df_dt.max().item()
                # print(f"Mean df_dt: {mean_df_dt}, Min df_dt: {min_df_dt}, Max df_dt: {max_df_dt}")

                # sys.exit(0)
                # Optionally, clip df_dt to prevent extreme values
                df_dt = torch.clamp(df_dt, min=-10, max=10)

                # Monothonicity Penalty for Decreasing Functions!!!!
                penalty = torch.mean(torch.relu(df_dt))
                #penalty = torch.mean(m(df_dt)) #Implement softplus to avoid 0 grads

                total_loss = loss + lambda_penalty * penalty
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(gp_model.parameters(), max_norm=1.0)

                optimizer.step()

                running_loss += total_loss.item() * inputs.size(0)
                running_data_loss += loss.item() * inputs.size(0)
                running_penalty_loss += penalty.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        epoch_data_loss = running_data_loss / len(train_dataset)
        epoch_penalty_loss = running_penalty_loss / len(train_dataset)

        total_losses.append(epoch_loss)
        epoch_data_losses.append(epoch_data_loss)
        epoch_penalty_losses.append(epoch_penalty_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Total GP Loss: {epoch_loss:.4f}, Data Loss: {epoch_data_loss:.4f}, Monotonicity Penalty: {epoch_penalty_loss:.4f}")

        # After the batch loop is complete, calculate df/dt statistics
        # Store df/dt values at selected epochs
        if (epoch + 1) in epochs_to_record:
            df_dt_all = []
            for inputs, _, _ in train_loader:
                inputs = inputs.to(device, non_blocking=True).clone().detach().requires_grad_(True)
                targets = targets.to(device)
                output = model_wrapper(inputs)
                mean_output = output.mean
                df_dx = torch.autograd.grad(
                    outputs=mean_output,
                    inputs=inputs,
                    grad_outputs=torch.ones_like(mean_output),
                    create_graph=False
                )[0]
                df_dt = df_dx[:, -1]
                df_dt_all.append(df_dt.detach().cpu())

            df_dt_all = torch.cat(df_dt_all)
            df_dt_mean = df_dt_all.mean().item()
            df_dt_std = df_dt_all.std().item()

            df_dt_means.append(df_dt_mean)
            df_dt_stds.append(df_dt_std)
            
            df_dt_values_over_epochs.append((epoch + 1, df_dt_all.numpy()))

    # ========================================
    # Analysis
    # ========================================
    # Loss Visualization 
    import matplotlib.pyplot as plt
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_losses, label='Total Loss')
    plt.plot(epochs, epoch_data_losses, label='Data Loss')
    plt.plot(epochs, epoch_penalty_losses, label='Monotonicity Penalty')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Components Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('monotonicsvdk_loss_components_'+str(lambda_penalty)+'.png',  dpi=300)
    plt.savefig('monotonicsvdk_loss_components_'+str(lambda_penalty)+'.svg',  dpi=300)

    # Gradient Statistics 
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_to_record, df_dt_means, label='Mean df/dt')
    plt.plot(epochs_to_record, df_dt_stds, label='Std df/dt')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Mean and Standard Deviation of df/dt Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('monotonicsvdk_df_dt_statistics_'+str(lambda_penalty)+'.png',  dpi=300)
    plt.savefig('monotonicsvdk_df_dt_statistics_'+str(lambda_penalty)+'.svg',  dpi=300)

    for epoch_num, df_dt_values in df_dt_values_over_epochs:
        plt.figure(figsize=(10, 6))
        plt.hist(df_dt_values, bins=50, alpha=0.7)
        plt.xlabel('df/dt')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of df/dt at Epoch {epoch_num}')
        plt.grid(True)
        plt.show()
        plt.savefig(f'df_dt_histogram_epoch_{epoch_num}.png',  dpi=300)


    # Save df/dt values over epochs
    with open('monotonicsvdk_df_dt_values_over_epochs_'+str(lambda_penalty)+'.pkl', 'wb') as f:
        pickle.dump(df_dt_values_over_epochs, f)

    # =======================================
    # Step 5: Evaluation
    # =======================================
    model_wrapper.eval()
    likelihood.eval()
    subjects_with_mismatch = []
    with torch.no_grad():
        predictions = []
        actuals = []
        roc_gt_all, roc_pred_all = [], []
        for inputs, targets, subject_id in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            subject_id = subject_id[0]
            output = model_wrapper(inputs)
            pred = likelihood(output)

            subject_mean = pred.mean.detach().cpu().numpy()
            subject_variance = pred.variance.detach().cpu().numpy()
            subject_std_dev = np.sqrt(subject_variance)

            subject_groundtruth = targets.detach().cpu().numpy()

            age = longitudinal_covariates[longitudinal_covariates['PTID'] == subject_id]['Age'].values

            subject_time = inputs[:, -1].detach().cpu().numpy().tolist()

            if len(age)!=len(subject_time):
                print('Length mismatch')
                print(len(age), len(subject_time))
                print(age)
                print(subject_time)
                subjects_with_mismatch.append(subject_id)
                continue 
            # assert len(age) == len(subject_time)

            # Interval 
            lower_bound = subject_mean - 1.96 * subject_std_dev
            upper_bound = subject_mean + 1.96 * subject_std_dev
            interval = np.mean(np.abs(upper_bound - lower_bound))
            
            # Coverage
            coverage = calculate_coverage(targets.detach().cpu().numpy(), lower_bound, upper_bound)

            # RoC Groundtruth
            roc_gt = calculate_rate_of_change(age, subject_groundtruth)

            # RoC Prediction
            roc_pred = calculate_rate_of_change(age, subject_mean)

            # RoC Deviation: Absolute difference between actual and predicted RoC
            roc_dev = np.abs(roc_gt - roc_pred)

            # Per Subject Metrics
            population_per_subject_metrics['id'].append(subject_id)
            population_per_subject_metrics['fold'].append(fold)
            population_per_subject_metrics['mae'].append(mean_absolute_error(targets.detach().cpu().numpy(), pred.mean.detach().cpu().numpy()))
            population_per_subject_metrics['mse'].append(mean_squared_error(targets.detach().cpu().numpy(), pred.mean.detach().cpu().numpy()))
            population_per_subject_metrics['coverage'].append(coverage)
            population_per_subject_metrics['interval'].append(interval)
            population_per_subject_metrics['roc_dev'].append(roc_dev)
            population_per_subject_metrics['gt_roc'].append(roc_gt)
            population_per_subject_metrics['pred_roc'].append(roc_pred)

            # Per Subject Monotonicity Evaluation
            temp_list = pred.mean.detach().cpu().numpy()
            #print("Subject samples: ", temp_list)
            monotonic = True
            check = 0
            timestep_list = [len(temp_list)]
            for i in range(len(temp_list)):
                if(check == 0):
                    max_num = temp_list[i]
                    check = 1
                else:
                    if(temp_list[i]>max_num):
                        monotonic = False
                        timestep_list.append(i)
                    else:
                        max_num = temp_list[i]

            # if(monotonic):
            #     print("Monotonicity is preserved throughout the sequence")
            # else:
            #     print("Monotonicity is not preserved")

            population_per_subject_metrics['monotonicity'].append(int(monotonic))
            population_per_subject_metrics['timesteps'].append(timestep_list)
            population_per_subject_metrics['sequence'].append(temp_list)
            population_per_subject_metrics['true_sequence'].append(targets.detach().cpu().numpy())
            population_per_subject_metrics['subject_time'].append(subject_time)


            # Subject Predictions 
            population_results['id'].extend([subject_id] * len(subject_mean))
            population_results['fold'].extend([fold] * len(subject_mean))
            population_results['score'].extend(subject_mean)
            population_results['y'].extend(subject_groundtruth)
            population_results['variance'].extend(subject_variance)
            population_results['time'].extend(subject_time)
            population_results['age'].extend(age)

            # Gather Total Results
            predictions.extend(pred.mean.detach().cpu().numpy())
            actuals.extend(targets.detach().cpu().numpy())
            roc_gt_all.append(roc_gt)
            roc_pred_all.append(roc_pred)

    # Calculate evaluation metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"Test R^2: {r2:.4f}")

    # Store the avegare metrics of the fold 
    population_fold_metrics['fold'].append(fold)
    population_fold_metrics['mse'].append(mse)
    population_fold_metrics['mae'].append(mae)
    population_fold_metrics['r2'].append(r2)
    population_fold_metrics['coverage'].append(np.mean(population_per_subject_metrics['coverage']))
    population_fold_metrics['interval'].append(np.mean(population_per_subject_metrics['interval']))
    population_fold_metrics['mean_roc_dev'].append(np.mean(population_per_subject_metrics['roc_dev']))

    # Assuming you have numpy arrays of predictions and targets
    metrics_ci = bootstrap_metrics(predictions, targets, n_bootstrap=1000, alpha=0.05)

    print("Confidence Intervals for Performance Metrics:")
    print(f"RMSE 95% CI: [{metrics_ci['rmse'][0]:.4f}, {metrics_ci['rmse'][1]:.4f}]")
    print(f"MAE 95% CI: [{metrics_ci['mae'][0]:.4f}, {metrics_ci['mae'][1]:.4f}]")
    print(f"R² 95% CI: [{metrics_ci['r2'][0]:.4f}, {metrics_ci['r2'][1]:.4f}]")

    # Save the GP model.
    # Todo: Save the whole state in case of further training 
    torch.save(gp_model.state_dict(), 'svdk_gp_monotonic_model_'+str(lambda_penalty)+'.pth')
    torch.save(likelihood.state_dict(), 'svdk_monotonic_likelihood'+str(lambda_penalty)+'.pth')

    # Set the model to evaluation mode
    model_wrapper.eval()
    likelihood.eval()

    print(f"Total subjects with mismatched lengths: {len(subjects_with_mismatch)}")
    print(f"Subjects with mismatched lengths: {subjects_with_mismatch}")

    # =======================================
    # Step 6: Check Monotonicity of Predictions (Per Sample)
    # =======================================

    total_samples = 0
    monotonic_samples = 0

    for inputs, targets, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True).clone().detach().requires_grad_(True)
        targets = targets.to(device)
        output = model_wrapper(inputs)
        mean_output = output.mean
        df_dx = torch.autograd.grad(
            outputs=mean_output,
            inputs=inputs,
            grad_outputs=torch.ones_like(mean_output),
            create_graph=False
        )[0]
        df_dt = df_dx[:, -1]

        # Count total samples
        batch_size = inputs.size(0)
        total_samples += batch_size

        # Count samples where df_dt <= 0
        monotonic_samples += (df_dt <= 0).sum().item()

    # Calculate percentage
    monotonicity_percentage = (monotonic_samples / total_samples) * 100
    print(f"Percentage of samples where monotonicity is achieved: {monotonicity_percentage:.2f}%")

    # Save the results
    population_results_df = pd.DataFrame(population_results)
    population_fold_metrics_df = pd.DataFrame(population_fold_metrics)
    population_per_subject_metrics_df = pd.DataFrame(population_per_subject_metrics)

    population_results_df.to_csv('svdk_monotonic_'+str(lambda_penalty)+'_population_results.csv', index=False)
    population_fold_metrics_df.to_csv('svdk_monotonic_'+str(lambda_penalty)+'_population_fold_metrics.csv', index=False)
    population_per_subject_metrics_df.to_csv('svdk_monotonic_'+str(lambda_penalty)+'_population_per_subject_metrics.csv', index=False)

if __name__ == "__main__":
    main()
