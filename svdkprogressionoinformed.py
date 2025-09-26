'''
Stochastic Variational Deep Kernel Regression with Monotonicity and Smoothness Constraints 
and Self-Supervised Progression Inform Metric Learning for the Latent Space.
'''
import numpy as np
import pandas as pd
import torch
import argparse
import pickle 
from torch.utils.data import Dataset
import gpytorch 
from functions import process_temporal_singletask_data  # Assuming you have this function defined
import json 
import random 
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn as nn 

#### MODELS #####
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

class DeepKernelGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, feature_extractor):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(DeepKernelGPModel, self).__init__(variational_strategy)

        self.feature_extractor = feature_extractor  # Feature extractor to be trained jointly
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#### DATA, DATASET AND DATALOADERS ####
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

class CognitiveDataset(Dataset):
    def __init__(self, inputs, targets, subject_ids, progression_rates):
        """
        inputs: NumPy array of input features, shape (num_samples, input_dim)
        targets: NumPy array of target values, shape (num_samples,)
        subject_ids: List or array of subject IDs, length num_samples
        progression_rates: NumPy array of progression rates, shape (num_samples,)
        """
        assert len(inputs) == len(targets) == len(subject_ids) == len(progression_rates), "Inputs, targets, subject_ids, and progression_rates must have the same length."

        self.inputs = torch.tensor(inputs, dtype=torch.float64)
        self.targets = torch.tensor(targets, dtype=torch.float64)
        self.subject_ids = subject_ids  # List or array of subject IDs
        self.progression_rates = torch.tensor(progression_rates, dtype=torch.float64)

        # Create a mapping from subject ID to indices
        self.subject_to_indices = {}
        for idx, subject_id in enumerate(self.subject_ids):
            if subject_id not in self.subject_to_indices:
                self.subject_to_indices[subject_id] = []
            self.subject_to_indices[subject_id].append(idx)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.subject_ids[idx], self.progression_rates[idx]

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

def collate_fn(batch):
    # 'batch' is a list of samples, where each sample is a tuple (input, target, subject_id, progression_rate)
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    subject_ids = [item[2] for item in batch]  # Keep as list
    progression_rates = torch.stack([item[3] for item in batch])

    return inputs, targets, subject_ids, progression_rates

##### PROGRESSION LOSS  #####
def compute_progression_rates(subject_ids, times, measurements):
    """
    Computes progression rates (slopes) for each subject.

    Args:
        subject_ids (list): List of subject IDs corresponding to each measurement.
        times (np.ndarray): Array of time points corresponding to each measurement.
        measurements (np.ndarray): Array of measurements.

    Returns:
        progression_rates (dict): Dictionary mapping subject IDs to their progression rates.
    """
    progression_rates = {}
    unique_subjects = np.unique(subject_ids)
    for subject in unique_subjects:
        idx = np.where(np.array(subject_ids) == subject)[0]
        time_points = times[idx]
        values = measurements[idx]
        if len(time_points) >= 2:
            # Fit a linear model
            slope, intercept = np.polyfit(time_points, values, 1)
            progression_rates[subject] = slope
        else:
            # Not enough data points to compute a slope
            progression_rates[subject] = 0.0  # Or handle as appropriate
    return progression_rates

def pairwise_loss(z, progression_rates, alpha=1.0):
    """
    Computes the pairwise loss between latent representations and progression rates.

    Parameters:
    - z: Tensor of latent representations, shape (batch_size, latent_dim)
    - progression_rates: Tensor of progression rates, shape (batch_size,)
    - alpha: Scaling factor to adjust the influence of progression rate differences

    Returns:
    - loss: Scalar tensor representing the pairwise loss
    """
    # Compute pairwise distances in latent space
    latent_distances = torch.cdist(z, z, p=2)  # Shape: (batch_size, batch_size)
    # Compute pairwise differences in progression rates
    progression_differences = torch.abs(progression_rates.unsqueeze(1) - progression_rates.unsqueeze(0))  # Shape: (batch_size, batch_size)
    # Desired distances in latent space
    desired_distances = alpha * progression_differences
    # Compute the loss
    loss = (latent_distances - desired_distances).pow(2)
    # Take the mean of the upper triangle (excluding diagonal) to avoid redundancy
    mask = torch.triu(torch.ones_like(loss), diagonal=1).bool()
    loss = loss[mask].mean()
    return loss


def main():
    fold = 0
    parser = argparse.ArgumentParser(description='Deep Regression for Neurodegeneration Prediction')
    # Data Parameters
    parser.add_argument("--experimentID", help="Indicates the Experiment Identifier", default='adniblsa')
    parser.add_argument("--file", help="Identifier for the data", default="subjectsamples_longclean_hmuse_adniblsa")
    parser.add_argument("--folder", type=int, default=2)
    parser.add_argument("--lambda_penalty", type=float, default=1)

    args = parser.parse_args()
    expID = args.experimentID
    file = args.file
    folder = args.folder
    lambda_penalty = args.lambda_penalty


    longitudinal_covariates = pd.read_csv('/home/cbica/Desktop/LongGPClustering/data' + str(2) + '/longitudinal_covariates_subjectsamples_longclean_hmuse_convs_' + 'adniblsa' +'.csv')
    longitudinal_covariates['Diagnosis'].replace([-1.0, 0.0, 1.0, 2.0], ['UKN', 'CN', 'MCI', 'AD'], inplace=True)

    population_results = {'id': [],'fold': [], 'score': [], 'y': [], 'variance': [], 'time': [], 'age': [] }
    population_fold_metrics = {'fold': [] , 'mse': [], 'mae': [], 'r2': [], 'coverage': [], 'interval': [], 'mean_roc_dev': []}
    population_per_subject_metrics = {'id': [], 'fold':[], 'mae': [], 'mse': [], 'coverage': [], 'interval': [], 'roc_dev': [], 'gt_roc': [], 'pred_roc': []}

    train_ids, test_ids = [], []
    # Load train IDs
    with open("/home/cbica/Desktop/LongGPClustering/data" + str(folder) + "/train_subject_" + expID + "_ids_hmuse" + str(fold) + ".pkl", "rb") as openfile:
        while True:
            try:
                train_ids.append(pickle.load(openfile))
            except EOFError:
                break

    # Load test IDs
    with open("/home/cbica/Desktop/LongGPClustering/data" + str(folder) + "/test_subject_" + expID + "_ids_hmuse" + str(fold) + ".pkl", "rb") as openfile:
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

    # Extract time points for train and test data
    train_times = train_x[:, -1].numpy()
    test_times = test_x[:, -1].numpy()

    # Convert tensors to numpy arrays
    train_subject_ids = np.array(corresponding_train_ids)
    test_subject_ids = np.array(corresponding_test_ids)
    train_measurements = train_y.numpy()
    test_measurements = test_y.numpy()

    # Compute progression rates for training data
    train_progression_rates_dict = compute_progression_rates(
        subject_ids=train_subject_ids,
        times=train_times,
        measurements=train_measurements
    )

    # Map progression rates to each data point in the training set
    train_progression_rates = np.array([train_progression_rates_dict[sid] for sid in train_subject_ids])

    # Normalize progression rates
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_progression_rates = scaler.fit_transform(train_progression_rates.reshape(-1, 1)).flatten()

    # Do the same for test data if needed (for consistency)
    test_progression_rates_dict = compute_progression_rates(
        subject_ids=test_subject_ids,
        times=test_times,
        measurements=test_measurements
    )
    test_progression_rates = np.array([test_progression_rates_dict[sid] for sid in test_subject_ids])
    test_progression_rates = scaler.transform(test_progression_rates.reshape(-1, 1)).flatten()

    batch_size = 64  # Adjust as needed
    train_sampler = SubjectBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    test_subject_sampler = TestSubjectBatchSampler(test_dataset, shuffle=False)

    # Create datasets including progression rates
    train_dataset = CognitiveDataset(inputs=train_x, targets=train_y, subject_ids=corresponding_train_ids, progression_rates=train_progression_rates)
    test_dataset = CognitiveDataset(inputs=test_x, targets=test_y, subject_ids=corresponding_test_ids, progression_rates=test_progression_rates)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_subject_sampler,
        collate_fn=collate_fn
    )


    # =======================================
    # Step 1: Train the Deep Regression Model
    # =======================================
    # Initialize the model
    feature_extractor = FeatureExtractorLatentConcatenation(input_dim, hidden_dim)
    model = RegressionNNLatentConcatenation(feature_extractor)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop for deep regression model
    num_epochs = 30  # Adjust as needed
    total_regression_loss = [] 
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
        # Step 4: Training Loop for GP Model with Pairwise Loss
        # =======================================
        # Adjusted Training Loop
        num_epochs = 200
        learning_rate = 1e-4  # Adjust learning rate as needed

        # Ensure that the feature extractor is in training mode
        feature_extractor_gp.train()

        optimizer = torch.optim.Adam([
        {'params': gp_model.parameters()},
        {'params': feature_extractor_gp.parameters()},
        {'params': likelihood.parameters()},
        ], lr=learning_rate)

        alpha = 1.0  # Scaling factor for progression rate differences
        lambda_pairwise = 1.0  # Weight for the pairwise loss term

        total_losses = []
        data_losses = []
        pairwise_losses = []

        for epoch in range(num_epochs):
        model_wrapper.train()
        likelihood.train()
        feature_extractor_gp.train()
        running_loss = 0.0
        running_data_loss = 0.0
        running_pairwise_loss = 0.0
        for inputs, targets, _, progression_rates in train_loader:
            optimizer.zero_grad()
            # Forward pass through feature extractor
            z = feature_extractor_gp(inputs)
            # Forward pass through GP model
            output = model_wrapper.gp_model(inputs)
            nll = -mll(output, targets)
            # Pairwise loss
            pw_loss = pairwise_loss(z, progression_rates, alpha=alpha)
            # Total loss
            total_loss = nll + lambda_pairwise * pw_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * inputs.size(0)
            running_data_loss += nll.item() * inputs.size(0)
            running_pairwise_loss += pw_loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        epoch_data_loss = running_data_loss / len(train_dataset)
        epoch_pairwise_loss = running_pairwise_loss / len(train_dataset)
        total_losses.append(epoch_loss)
        data_losses.append(epoch_data_loss)
        pairwise_losses.append(epoch_pairwise_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Total GP Loss: {epoch_loss:.4f}, Data Loss: {epoch_data_loss:.4f}, Pairwise Loss: {epoch_pairwise_loss:.4f}")


    # =======================================
    # Step 5: Evaluation
    # =======================================
    model_wrapper.eval()
    likelihood.eval()
    feature_extractor_gp.eval()
    subjects_with_mismatch = []
    with torch.no_grad():
        predictions = []
        actuals = []
        roc_gt_all, roc_pred_all = [], []
        for inputs, targets, subject_id, progression_rates in test_loader:
            subject_id = subject_id[0]
            output = model_wrapper(inputs)
            pred = likelihood(output)

            # ... (rest of the code remains the same)
