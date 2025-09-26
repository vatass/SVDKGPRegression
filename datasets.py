### Datasets ###
import torch 
from torch.utils.data import Dataset

#### Train Dataset ####
class TrainDataset(Dataset):
    def __init__(self, features, labels, ids):
        """
        Initialize the dataset with features, labels, and corresponding sample ids.
        
        Args:
            features (np.array): Input features for the model.
            labels (np.array): Labels for the input features.
            ids (list or np.array): Identifiers for each sample.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.ids = ids
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.ids[index]

### Test Dataset ###
class TestDataset(Dataset):
    def __init__(self, features, labels, ids):
        """
        Initialize the dataset with features, labels, and corresponding sample ids.
        
        Args:
            features (np.array): Input features for the model.
            labels (np.array): Labels for the input features.
            ids (list or np.array): Identifiers for each sample.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.ids = ids
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.ids[index]
    
    def get_subject_data(self, subject_id):
        """
        Get the data corresponding to a specific subject ID.
        
        Args:
            subject_id (int or str): The subject ID for which to retrieve data.
        
        Returns:
            tuple: (first_data_point, remaining_data_points), where:
                   - first_data_point is a tuple of (features, label, id) for the first data point.
                   - remaining_data_points is a list of tuples (features, label, id) for the rest of the data points.
        """
        # Find indices for the specific subject
        indices = [i for i, x in enumerate(self.ids) if x == subject_id]
        
        if len(indices) == 0:
            raise ValueError(f"No data found for subject ID {subject_id}")
        
        # Split the first data point from the rest
        subject_baseline_index = indices[0]
        subject_unseen_indices = indices[1:]
        
        baseline_data = (self.features[subject_baseline_index, :], self.labels[subject_baseline_index], self.ids[subject_baseline_index])
        unseen_data = [(self.features[i], self.labels[i], self.ids[i]) for i in subject_unseen_indices]
        
        # the unseen data should be stacked in a single tensor
        unseen_data = (torch.stack([x[0] for x in unseen_data]), torch.stack([x[1] for x in unseen_data]), [x[2] for x in unseen_data])

        return baseline_data, unseen_data
