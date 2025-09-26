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
import json
import pickle
import argparse

# Set default dtype to double precision
torch.set_default_dtype(torch.float64)

# Step 1: Define the CognitiveDataset class
class CognitiveDataset(Dataset):
    def __init__(self, inputs, targets, progression_labels, subject_ids):
        """
        inputs: NumPy array or tensor of input features, shape (num_samples, input_dim)
        targets: NumPy array or tensor of regression targets, shape (num_samples, num_outputs)
        progression_labels: NumPy array or tensor of classification labels (0 or 1), shape (num_samples,)
        subject_ids: List or array of subject IDs, length num_samples
        """
        assert len(inputs) == len(targets) == len(progression_labels) == len(subject_ids), \
            "Inputs, targets, progression_labels, and subject_ids must have the same length."

        self.inputs = torch.tensor(inputs, dtype=torch.float64)
        self.targets = torch.tensor(targets, dtype=torch.float64)  # Shape: [num_samples, num_outputs]
        self.progression_labels = torch.tensor(progression_labels, dtype=torch.float64)
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
        return self.inputs[idx], self.targets[idx], self.progression_labels[idx], self.subject_ids[idx]

# Step 2: Define the FeatureExtractor class
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        features = self.relu(self.fc1(x))
        features = self.dropout(features)
        return features

# Step 3: Define the Multitask GP Regression Model
class MultitaskDeepKernelGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks, feature_extractor):
        batch_shape = torch.Size([num_tasks])  # For multitask GPs
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=batch_shape
        )
        variational_strategy = gpytorch.variational.InducingPointVariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
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

# Step 4: Define the GP Classifier
class GPClassifier(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, feature_extractor):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPClassifier, self).__init__(variational_strategy)
        self.feature_extractor = feature_extractor  # Shared feature extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Step 5: Define the Multi-task Model Wrapper
class MultiTaskGPModel(nn.Module):
    def __init__(self, gp_regression_model, regression_likelihood, gp_classification_model, classification_likelihood):
        super(MultiTaskGPModel, self).__init__()
        self.gp_regression_model = gp_regression_model
        self.regression_likelihood = regression_likelihood
        self.gp_classification_model = gp_classification_model
        self.classification_likelihood = classification_likelihood

    def forward(self, x):
        # GP Regression Output
        gp_regression_output = self.gp_regression_model(x)
        # GP Classification Output
        gp_classification_output = self.gp_classification_model(x)
        return gp_regression_output, gp_classification_output

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

# Step 8: Main function
def main():
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
    with open(f"/path/to/data{folder}/train_subject_{expID}_ids_hmuse{fold}.pkl", "rb") as openfile:
        train_ids = pickle.load(openfile)
    with open(f"/path/to/data{folder}/test_subject_{expID}_ids_hmuse{fold}.pkl", "rb") as openfile:
        test_ids = pickle.load(openfile)

    print('Train IDs:', len(train_ids))
    print('Test IDs:', len(test_ids))

    # Load and preprocess data
    train_x, train_y, train_progression_labels, test_x, test_y, test_progression_labels, \
    corresponding_train_ids, corresponding_test_ids = load_and_preprocess_data(
        folder=folder,
        file=file,
        train_ids=train_ids,
        test_ids=test_ids
    )

    # Ensure train_y and test_y have shape [num_samples, num_outputs]
    num_outputs = train_y.shape[1]

    # Standardize temporal variable (assuming it's the last column)
    temporal_index = -1
    scaler_time = StandardScaler()
    train_x[:, temporal_index] = scaler_time.fit_transform(train_x[:, temporal_index].reshape(-1, 1)).flatten()
    test_x[:, temporal_index] = scaler_time.transform(test_x[:, temporal_index].reshape(-1, 1)).flatten()

    # Create datasets
    train_dataset = CognitiveDataset(inputs=train_x, targets=train_y, progression_labels=train_progression_labels, subject_ids=corresponding_train_ids)
    test_dataset = CognitiveDataset(inputs=test_x, targets=test_y, progression_labels=test_progression_labels, subject_ids=corresponding_test_ids)

    # Create data loaders
    batch_size = 64  # Adjust as needed
    train_sampler = SubjectBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine input dimension
    input_dim = train_x.shape[1]
    hidden_dim = 128  # Adjust as needed

    # Initialize the shared feature extractor
    feature_extractor = FeatureExtractor(input_dim, hidden_dim)

    # Prepare the inducing points
    unique_train_subject_ids = list(set(corresponding_train_ids))
    selected_subject_ids = random.sample(unique_train_subject_ids, 200)  # Adjust the number as needed
    inducing_points = select_inducing_points(train_x.numpy(), corresponding_train_ids, selected_subject_ids=selected_subject_ids, num_points_per_subject=3)

    # Ensure inducing points are in torch.float64
    inducing_points = inducing_points.double()

    # Initialize GP Regression Model and Likelihood
    gp_regression_model = MultitaskDeepKernelGPModel(inducing_points, num_tasks=num_outputs, feature_extractor=feature_extractor)
    regression_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_outputs)

    # Initialize GP Classification Model and Likelihood
    gp_classification_model = GPClassifier(inducing_points, feature_extractor)
    classification_likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    # Convert models and likelihoods to double precision
    gp_regression_model = gp_regression_model.double()
    regression_likelihood = regression_likelihood.double()
    gp_classification_model = gp_classification_model.double()
    classification_likelihood = classification_likelihood.double()

    # Multi-task model wrapper
    model_wrapper = MultiTaskGPModel(
        gp_regression_model,
        regression_likelihood,
        gp_classification_model,
        classification_likelihood
    )

    # Define loss functions
    mll_regression = gpytorch.mlls.VariationalELBO(regression_likelihood, gp_regression_model, num_data=len(train_dataset))
    mll_classification = gpytorch.mlls.VariationalELBO(classification_likelihood, gp_classification_model, num_data=len(train_dataset))

    # Set up the optimizer
    optimizer = torch.optim.Adam([
        {'params': gp_regression_model.parameters()},
        {'params': regression_likelihood.parameters()},
        {'params': gp_classification_model.parameters()},
        {'params': classification_likelihood.parameters()},
    ], lr=1e-3)

    # Training Loop
    num_epochs = 100  # Adjust as needed
    classification_loss_weight = 1.0  # Adjust as needed

    for epoch in range(num_epochs):
        model_wrapper.train()
        regression_likelihood.train()
        classification_likelihood.train()
        running_loss = 0.0

        for inputs, targets, prog_labels, _ in train_loader:
            optimizer.zero_grad()
            gp_regression_output, gp_classification_output = model_wrapper(inputs)

            # Regression Loss
            loss_regression = -mll_regression(gp_regression_output, targets)

            # Classification Loss
            loss_classification = -mll_classification(gp_classification_output, prog_labels)

            # Total Loss
            total_loss = loss_regression + classification_loss_weight * loss_classification

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {epoch_loss:.4f}")

    # Evaluation
    model_wrapper.eval()
    regression_likelihood.eval()
    classification_likelihood.eval()
    with torch.no_grad():
        regression_predictions = [[] for _ in range(num_outputs)]
        regression_actuals = [[] for _ in range(num_outputs)]
        classification_predictions = []
        classification_actuals = []

        for inputs, targets, prog_labels, _ in test_loader:
            gp_regression_output, gp_classification_output = model_wrapper(inputs)

            # Regression Predictions
            pred_regression = regression_likelihood(gp_regression_output)
            mean_pred = pred_regression.mean  # Shape: [batch_size, num_outputs]

            for i in range(num_outputs):
                regression_predictions[i].extend(mean_pred[:, i].numpy())
                regression_actuals[i].extend(targets[:, i].numpy())

            # Classification Predictions
            pred_classification = classification_likelihood(gp_classification_output)
            class_probs = pred_classification.mean.numpy()
            class_preds = (class_probs >= 0.5).astype(int)
            classification_predictions.extend(class_preds)
            classification_actuals.extend(prog_labels.numpy())

        # Regression Metrics for Each Output
        for i in range(num_outputs):
            mse = mean_squared_error(regression_actuals[i], regression_predictions[i])
            mae = mean_absolute_error(regression_actuals[i], regression_predictions[i])
            print(f"Output {i+1} - Test MSE: {mse:.4f}, MAE: {mae:.4f}")

        # Classification Metrics
        accuracy = accuracy_score(classification_actuals, classification_predictions)
        f1 = f1_score(classification_actuals, classification_predictions)
        print(f"Classification Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    # Optionally, save the models
    torch.save(gp_regression_model.state_dict(), 'gp_regression_model.pth')
    torch.save(regression_likelihood.state_dict(), 'regression_likelihood.pth')
    torch.save(gp_classification_model.state_dict(), 'gp_classification_model.pth')
    torch.save(classification_likelihood.state_dict(), 'classification_likelihood.pth')

# Helper function to load and preprocess data
def load_and_preprocess_data(folder, file, train_ids, test_ids):
    # Load your data
    with open('/path/to/roi_to_idx.json') as f:
        roi_to_idx = json.load(f)

    index_to_roi = {v: k for k, v in roi_to_idx.items()}

    # Load your data
    datasamples = pd.read_csv(f'/path/to/data{folder}/{file}.csv')

    # Set up the train/test data
    train_data = datasamples[datasamples['PTID'].isin(train_ids)]
    test_data = datasamples[datasamples['PTID'].isin(test_ids)]

    # Extract inputs and targets
    train_x = np.stack(train_data['X'].values)
    train_y_all = np.stack(train_data['Y'].values)
    test_x = np.stack(test_data['X'].values)
    test_y_all = np.stack(test_data['Y'].values)

    # Corresponding subject IDs
    corresponding_train_ids = train_data['PTID'].tolist()
    corresponding_test_ids = test_data['PTID'].tolist()

    # Progression labels
    train_progression_labels = train_data['Progressor'].values
    test_progression_labels = test_data['Progressor'].values

    # Indices of the target variables
    target_variables = ['Hippocampus', 'SPARE_AD', 'SPARE_BA', 'MMSE', 'ADAS-COG-13']
    target_indices = []
    for var in target_variables:
        if var in roi_to_idx:
            idx = roi_to_idx[var]
        else:
            if var in train_data.columns:
                idx = train_data.columns.get_loc(var)
            else:
                raise ValueError(f"Variable {var} not found in data.")
        target_indices.append(idx)

    # Extract the targets
    train_y = train_y_all[:, target_indices]
    test_y = test_y_all[:, target_indices]

    return train_x, train_y, train_progression_labels, test_x, test_y, test_progression_labels, corresponding_train_ids, corresponding_test_ids

if __name__ == "__main__":
    main()
