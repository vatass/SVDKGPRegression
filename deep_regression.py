# deep_regression.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
import pandas as pd
import json
from functions import process_temporal_singletask_data  # Assuming you have this function defined in a separate file

# Step 1: Define the Dataset class
from torch.utils.data import Dataset

class CognitiveDataset(Dataset):
    def __init__(self, inputs, targets, subject_ids):
        """
        inputs: NumPy array of input features, shape (num_samples, input_dim)
        targets: NumPy array of target values, shape (num_samples,)
        subject_ids: List or array of subject IDs, length num_samples
        """
        assert len(inputs) == len(targets) == len(subject_ids), "Inputs, targets, and subject_ids must have the same length."
        
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
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
        # Return inputs, targets, and subject IDs
        return self.inputs[idx], self.targets[idx], self.subject_ids[idx]



# Step 2: Define the model
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

# Custom Batch Sampler to batch data per subject
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


def main():

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, Sampler
    import numpy as np
    import random
    import pandas as pd
    import pickle
    import argparse

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

    # Assuming train_x, train_y, and corresponding_train_ids are defined and aligned
    train_dataset = CognitiveDataset(inputs=train_x, targets=train_y, subject_ids=corresponding_train_ids)

    # Similarly for the test dataset
    test_dataset = CognitiveDataset(inputs=test_x, targets=test_y, subject_ids=corresponding_test_ids)

    batch_size = 32  # Adjust as needed
    # Create DataLoaders
    train_sampler = SubjectBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    num_epochs = 10  # Adjust as needed
    hidden_dim = 128  # Adjust based on your data
   # Determine input dimension
    input_dim = train_x.shape[1]
    hidden_dim = 128  # Adjust based on your data

    # Initialize the model
    feature_extractor = FeatureExtractor(input_dim, hidden_dim)
    model = RegressionNN(feature_extractor)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 100  # Adjust as needed
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets, subject_ids in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # After the training loop
    # Save the entire model
    torch.save(model.state_dict(), 'deep_regression_model.pth')

    # Or save just the feature extractor
    torch.save(feature_extractor.state_dict(), 'feature_extractor.pth')


    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = []
        actuals = []
        for inputs, targets, subject_ids in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy())
            actuals.extend(targets.numpy())

    # Calculate evaluation metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")

if __name__ == "__main__":
    main()