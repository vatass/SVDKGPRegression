# svdk_regression_multitask.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
import pandas as pd
import gpytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from functions import process_temporal_singletask_data
import json
import pickle
import argparse
import os
# Set default dtype to double precision
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

# Step 2: Define the FeatureExtractor class
# Separate Processing and Addition
# Feature Extractor: Latent Concatenation of Time and Imaging Features
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc1(x))

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
    def __init__(self, feature_extractor, output_dim=4):
        super(RegressionNNLatentConcatenation, self).__init__()
        self.feature_extractor = feature_extractor

        # The output dimension of feature_extractor is hidden_dim
        self.fc_out = nn.Linear(feature_extractor.hidden_dim, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.fc_out(features)
        return output


# Step 3: Define the Multitask GP Regression Model
class MultitaskDeepKernelGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks, feature_extractor):
        batch_shape = torch.Size([num_tasks])  # For multitask GPs
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=batch_shape
        )
        base_variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            self, base_variational_strategy, num_tasks=batch_shape
        )
        super(MultitaskDeepKernelGPModel, self).__init__(variational_strategy)

        self.feature_extractor = feature_extractor  # Shared feature extractor
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
            batch_shape=batch_shape
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Step 5: Define the GPModelWrapper
class GPModelWrapper(nn.Module):
    def __init__(self, gp_model, likelihood):
        super(GPModelWrapper, self).__init__()
        self.gp_model = gp_model
        self.likelihood = likelihood

    def forward(self, x):
        return self.gp_model(x)

# Step 6: SubjectBatchSampler remains the same
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

# Step 7: Define the select_inducing_points function
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
            selected_values = selected_values.astype(np.float64)
            inducing_points_list.append(selected_values)
        else:
            print(f"Warning: No inducing points for subject {subject_id}.")

    if inducing_points_list:
        inducing_points_array = np.vstack(inducing_points_list)
        inducing_points = torch.tensor(inducing_points_array, dtype=torch.float64)
    else:
        raise ValueError("No inducing points were selected. Check your data and selection criteria.")

    return inducing_points

def load_and_preprocess_data(folder, file, train_ids, test_ids):
    f = open('/home/cbica/Desktop/LongGPClustering/roi_to_idx.json')
    roi_to_idx = json.load(f)

    index_to_roi = {v: k for k, v in roi_to_idx.items()}

    # Load your data
    
    datasamples = pd.read_csv('/home/cbica/Desktop/SVDKRegression/multitask_neuroimaging_biomarkers_allstudies.csv')

    # Set up the train/test data
    train_x = datasamples[datasamples['PTID'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['PTID'].isin(train_ids)]['Y']
    test_x = datasamples[datasamples['PTID'].isin(test_ids)]['X']
    test_y = datasamples[datasamples['PTID'].isin(test_ids)]['Y']

    # Corresponding subject IDs
    corresponding_train_ids = datasamples[datasamples['PTID'].isin(train_ids)]['PTID'].tolist()
    corresponding_test_ids = datasamples[datasamples['PTID'].isin(test_ids)]['PTID'].tolist()

    # Process the data
    train_x, train_y, test_x, test_y = process_temporal_singletask_data(
        train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, test_ids=test_ids
    )

    # Convert tensors to numpy arrays
    train_x = train_x.numpy()
    train_y = train_y.numpy()
    test_x = test_x.numpy()
    test_y = test_y.numpy()

    print("Shape of train_x, ", train_x.shape)
    print("Shape of train_y, ", train_y.shape)
    print("Shape of test_x, ", test_x.shape)
    print("Shape of test_y, ", test_y.shape)

    return train_x, train_y, test_x, test_y, corresponding_train_ids, corresponding_test_ids

def collate_fn(batch):
    # 'batch' is a list of samples, where each sample is a tuple (input, target, subject_id)
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    subject_ids = [item[2] for item in batch]  # Keep as list or convert to tensor if numeric
    
    return inputs, targets, subject_ids

# Step 8: Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fold = 0
    parser = argparse.ArgumentParser(description='Multi-output GP Regression and Classification for Neurodegeneration Prediction')
    # Data Parameters
    parser.add_argument("--experimentID", help="Indicates the Experiment Identifier", default='adniblsa')
    parser.add_argument("--file", help="Identifier for the data", default="subjectsamples_longclean_hmuse_adniblsa")
    parser.add_argument("--folder", type=int, default=2)
    args = parser.parse_args()
    expID = args.experimentID
    file = args.file
    folder = args.folder

    
    # Load train and test IDs
    train_ids, test_ids = [], []
    with (open("/home/cbica/Desktop/DKGP/data/train_subject_allstudies_ids_mmse" + str(fold) +  ".pkl", "rb")) as openfile:
        while True:
            try:
                train_ids.append(pickle.load(openfile))
            except EOFError:
                break 
    
    with (open("/home/cbica/Desktop/DKGP/data/test_subject_allstudies_ids_mmse" + str(fold) +  ".pkl", "rb")) as openfile:
        while True:
            try:
                test_ids.append(pickle.load(openfile))
            except EOFError:
                break

    train_ids = train_ids[0]
    test_ids = test_ids[0]

    print('Train IDs:', len(train_ids))
    print('Test IDs:', len(test_ids))

    # Load and preprocess data
    train_x, train_y, test_x, test_y, \
    corresponding_train_ids, corresponding_test_ids = load_and_preprocess_data(
        folder=folder,
        file=file,
        train_ids=train_ids,
        test_ids=test_ids
    )
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

    print("Train x shape :", train_x.shape)
    print("Train y shape :", train_y.shape)

    #How many tasks we have

    num_outputs = train_y.shape[1]

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

    # Determine input dimension
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
    output_file = "./multitask_trials"

    os.makedirs(output_file, exist_ok=True)
    print(f"Output directory {output_file} created")

    from pathlib import Path
    monotonicity_results = Path(f"{output_file}/results.txt")
    monotonicity_results.touch(exist_ok=True)
    torch.save(feature_extractor.state_dict(), '{}/multitask_feature_extractor_latentconcatenation.pth'.format(output_file))

    # Visualize the training loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(total_regression_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()
    plt.savefig('{}/regression_training_loss.png'.format(output_file),  dpi=300)
    plt.savefig('{}/regression_training_loss.svg'.format(output_file),  dpi=300)

    # =======================================
    # Step 2: Load Feature Extractor for GP Model
    # =======================================
    # Re-initialize the feature extractor and load the saved parameters
    #Create output folder for runs
    

    feature_extractor_gp = FeatureExtractorLatentConcatenation(input_dim, hidden_dim).to(device)
    feature_extractor_gp.load_state_dict(torch.load('{}/multitask_feature_extractor_latentconcatenation.pth'.format(output_file)))
    feature_extractor_gp.eval()

    # Prepare the inducing points
    unique_train_subject_ids = list(set(corresponding_train_ids))
    selected_subject_ids = random.sample(unique_train_subject_ids, 200)  # Adjust the number as needed
    inducing_points = select_inducing_points(train_x, corresponding_train_ids, selected_subject_ids=selected_subject_ids, num_points_per_subject=3)

    # Ensure inducing points are in torch.float64
    inducing_points = inducing_points.double().to(device)

    # Initialize GP Regression Model and Likelihood
    gp_regression_model = MultitaskDeepKernelGPModel(inducing_points, num_tasks=num_outputs, feature_extractor=feature_extractor_gp)
    regression_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_outputs)


    # Convert models and likelihoods to double precision
    gp_regression_model = gp_regression_model.double()
    regression_likelihood = regression_likelihood.double()
    model_wrapper = GPModelWrapper(gp_regression_model, regression_likelihood).to(device)
    # Define loss functions
    mll_regression = gpytorch.mlls.VariationalELBO(regression_likelihood, gp_regression_model, num_data=len(train_dataset))

    # Set up the optimizer
    optimizer = torch.optim.Adam([
        {'params': gp_regression_model.parameters()},
        {'params': regression_likelihood.parameters()}
    ], lr=1e-3)

    # Training Loop
    num_epochs = 100  # Adjust as neededd

    for epoch in range(num_epochs):
        model_wrapper.train()
        regression_likelihood.train()
        running_loss = 0.0

        for inputs, targets, _ in train_loader:
            print(inputs.shape)
            print(targets.shape)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            gp_regression_output = model_wrapper(inputs)

            # Regression Loss
            loss_regression = -mll_regression(gp_regression_output, targets)

            # Total Loss
            total_loss = loss_regression

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {epoch_loss:.4f}")

    # Evaluation

    model_wrapper.eval()
    regression_likelihood.eval()
    with torch.no_grad():
        regression_predictions = [[] for _ in range(num_outputs)]
        regression_actuals = [[] for _ in range(num_outputs)]


        for inputs, targets, _ in test_loader:
            gp_regression_output = model_wrapper(inputs)

            # Regression Predictions
            pred_regression = regression_likelihood(gp_regression_output)
            mean_pred = pred_regression.mean  # Shape: [batch_size, num_outputs]

            for i in range(num_outputs):
                regression_predictions[i].extend(mean_pred[:, i].numpy())
                regression_actuals[i].extend(targets[:, i].numpy())
        # Regression Metrics for Each Output
        for i in range(num_outputs):
            mse = mean_squared_error(regression_actuals[i], regression_predictions[i])
            mae = mean_absolute_error(regression_actuals[i], regression_predictions[i])
            print(f"Output {i+1} - Test MSE: {mse:.4f}, MAE: {mae:.4f}")


    # Optionally, save the models
    torch.save(gp_regression_model.state_dict(), 'gp_regression_model.pth')
    torch.save(regression_likelihood.state_dict(), 'regression_likelihood.pth')

if __name__ == "__main__":
    main()
