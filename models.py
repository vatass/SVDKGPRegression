import torch
import torch.nn as nn
import gpytorch
import math


#### Feature Extractor that works great on its own as a Regressor #####
class MLP(nn.Module):
    def __init__(self, input_dim, gp_hidden_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc5 = nn.Linear(32, gp_hidden_dim)  # Output 64 features for GP
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.dropout(self.relu4(self.fc4(x)))   
        x = self.fc5(x)
        return x

class MLP_Reduced(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(MLP_Reduced, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, output_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.LeakyReLU()  # or nn.ELU()

    def forward(self, x):
        x1 = self.dropout(self.activation(self.bn1(self.fc1(x))))
        x2 = self.dropout(self.activation(self.bn2(self.fc2(x1))))
        x4 = self.fc4(x2)
        return x4

import torch
import gpytorch
import torch.nn as nn


#### SVDK - Independent GPs on each Latent Dimension ####
class AdditiveDKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, gp_hidden_dim, input_dim, num_inducing):
        super(AdditiveDKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = AdditiveGaussianProcessLayer(num_dim=gp_hidden_dim, num_inducing=num_inducing,  input_dim=input_dim)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
        self.mixing_weights = nn.Parameter(torch.ones(gp_hidden_dim))

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        features = features.unsqueeze(-1)  # Add extra dimension for GP input
        gp_output = self.gp_layer(features)
        mixed_output = (gp_output.mean * self.mixing_weights).sum(dim=-1)
        return mixed_output

    def prediction(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        features = features.unsqueeze(-1)  # Add extra dimension for GP input
        gp_output = self.gp_layer(features)
        
        # Calculate mean
        mean = (gp_output.mean * self.mixing_weights).sum(dim=-1)
        
        # Calculate variance
        variance = (gp_output.variance * self.mixing_weights**2).sum(dim=-1)
        
        return gpytorch.distributions.Normal(mean, variance.sqrt())

class AdditiveGaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, num_inducing=500, input_dim=1):
        # Create inducing points
        inducing_points = torch.randn(num_inducing, input_dim)

        # Create a variational distribution for each output dimension
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([num_dim])
        )

        # Wrap the base variational strategy with IndependentMultitaskVariationalStrategy
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_dim
        )

        super().__init__(variational_strategy)

        # Use a different kernel for each output dimension
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_dim]))
        )
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_dim]))

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


###### A Single GP on the whole Latent Dimension #######
class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, num_inducing=500):
        inducing_points = torch.randn(num_inducing, num_dim)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=num_dim))
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, gp_hidden_dim, inducing_points):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=gp_hidden_dim, num_inducing=inducing_points)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        gp_output = self.gp_layer(features)
        return gp_output.mean

    def prediction(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        return self.gp_layer(features)


######################################################################################



#### Advanced Models ####

# ### 1. Progression Status Informed Model ###
# train_loader = [] 
# import torch
# import torch.nn as nn
# import gpytorch

# class ProgressorClassifier(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(latent_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
    
#     def forward(self, x):
#         return torch.sigmoid(self.classifier(x))

# class InformedFeatureExtractor(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super().__init__()
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, latent_dim)
#         )
#         self.progressor_classifier = ProgressorClassifier(latent_dim)
    
#     def forward(self, x):
#         latent = self.feature_extractor(x)
#         prog_prob = self.progressor_classifier(latent)
#         return latent, prog_prob

# class InformedGPLayer(gpytorch.models.ApproximateGP):
#     def __init__(self, num_inducing, input_dim):
#         variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing)
#         variational_strategy = gpytorch.variational.VariationalStrategy(self, torch.randn(num_inducing, input_dim), variational_distribution, learn_inducing_locations=True)
#         super().__init__(variational_strategy)
#         self.mean_module = gpytorch.means.LinearMean(input_dim)
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))
    
#     def forward(self, x, prog_prob):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         # Adjust the covariance based on the progressor probability
#         covar_x = covar_x * (1 + prog_prob.unsqueeze(1))
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# class InformedDKLModel(gpytorch.Module):
#     def __init__(self, feature_extractor, gp_layer):
#         super().__init__()
#         self.feature_extractor = feature_extractor
#         self.gp_layer = gp_layer
    
#     def forward(self, x):
#         latent, prog_prob = self.feature_extractor(x)
#         return self.gp_layer(latent, prog_prob)

# # Initialize the model
# input_dim = 145 + 5  # 145 imaging features + 5 demographic/genetic features
# latent_dim = 32
# num_inducing = 100

# feature_extractor = InformedFeatureExtractor(input_dim, latent_dim)
# gp_layer = InformedGPLayer(num_inducing, latent_dim)
# model = InformedDKLModel(feature_extractor, gp_layer)
# likelihood = gpytorch.likelihoods.GaussianLikelihood()

# # Training loop
# def train(model, likelihood, train_loader, optimizer, mll, epochs):
#     for epoch in range(epochs):
#         model.train()
#         likelihood.train()
#         epoch_loss = 0
#         for batch_x, batch_y, batch_prog in train_loader:
#             optimizer.zero_grad()
#             latent, prog_prob = model.feature_extractor(batch_x)
#             output = model.gp_layer(latent, prog_prob)
            
#             # GP regression loss
#             loss = -mll(output, batch_y)
            
#             # Binary classification loss for progressor prediction
#             prog_loss = nn.BCELoss()(prog_prob, batch_prog.float())
            
#             # Combine losses
#             total_loss = loss + prog_loss
#             total_loss.backward()
#             optimizer.step()
#             epoch_loss += total_loss.item()
#         print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}')

# # Use the model
# optimizer = torch.optim.Adam([
#     {'params': model.parameters()},
#     {'params': likelihood.parameters()},
# ], lr=0.01)

# mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

# train(model, likelihood, train_loader, optimizer, mll, epochs=100)

# # Prediction function
# def predict(model, x):
#     model.eval()
#     with torch.no_grad():
#         latent, prog_prob = model.feature_extractor(x)
#         output = model.gp_layer(latent, prog_prob)
#         return output.mean, prog_prob

# # Example usage
# test_x = torch.randn(10, input_dim)  # 10 test samples
# predicted_y, predicted_prog_prob = predict(model, test_x)


# import torch
# import torch.nn as nn
# import gpytorch

# class SNPAttentionGenerator(nn.Module):
#     def __init__(self, num_snps, num_imaging_features, hidden_dim=64):
#         super().__init__()
#         self.embedding = nn.Embedding(3, 4)  # 3 for possible genotypes (0, 1, 2), embedding dim 4
#         self.fc1 = nn.Linear(num_snps * 4, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, num_imaging_features)
    
#     def forward(self, x):
#         # x shape: (batch_size, num_snps)
#         x = self.embedding(x)
#         x = x.view(x.size(0), -1)  # Flatten the embeddings
#         x = torch.relu(self.fc1(x))
#         attention = torch.sigmoid(self.fc2(x))  # Sigmoid to get values between 0 and 1
#         return attention

# class InformedFeatureExtractor(nn.Module):
#     def __init__(self, input_dim, latent_dim, num_snps, num_imaging_features):
#         super().__init__()
#         self.num_imaging_features = num_imaging_features
#         self.snp_attention = SNPAttentionGenerator(num_snps, num_imaging_features)
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, latent_dim)
#         )
#         self.progressor_classifier = nn.Sequential(
#             nn.Linear(latent_dim + 1, 64),  # +1 for time
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )
    
#     def forward(self, x, time, snp_data):
#         # Generate attention map from SNP data
#         attention = self.snp_attention(snp_data)
        
#         # Apply attention to imaging features
#         imaging_features = x[:, :self.num_imaging_features]
#         attended_imaging = imaging_features * attention
        
#         # Concatenate attended imaging features with other features
#         other_features = x[:, self.num_imaging_features:]
#         x_attended = torch.cat([attended_imaging, other_features], dim=1)
        
#         # Extract latent features
#         latent = self.feature_extractor(x_attended)
        
#         # Predict progression probability
#         prog_input = torch.cat([latent, time.unsqueeze(1)], dim=1)
#         prog_prob = torch.sigmoid(self.progressor_classifier(prog_input))
        
#         return latent, prog_prob, attention

# class InformedGPLayer(gpytorch.models.ApproximateGP):
#     def __init__(self, num_inducing, input_dim):
#         variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing)
#         variational_strategy = gpytorch.variational.VariationalStrategy(
#             self, torch.randn(num_inducing, input_dim + 1),  # +1 for time
#             variational_distribution, learn_inducing_locations=True
#         )
#         super().__init__(variational_strategy)
#         self.mean_module = gpytorch.means.LinearMean(input_dim + 1)
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim + 1))
    
#     def forward(self, x, time):
#         x_time = torch.cat([x, time.unsqueeze(1)], dim=1)
#         mean_x = self.mean_module(x_time)
#         covar_x = self.covar_module(x_time)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# class InformedDKLModel(gpytorch.Module):
#     def __init__(self, feature_extractor, gp_layer):
#         super().__init__()
#         self.feature_extractor = feature_extractor
#         self.gp_layer = gp_layer
    
#     def forward(self, x, time, snp_data):
#         latent, prog_prob, attention = self.feature_extractor(x, time, snp_data)
#         return self.gp_layer(latent, time), prog_prob, attention

# # Initialize the model
# input_dim = 145 + 5  # 145 imaging features + 5 demographic/genetic features
# latent_dim = 32
# num_inducing = 100
# num_snps = 1000  # Adjust based on your SNP data
# num_imaging_features = 145

# feature_extractor = InformedFeatureExtractor(input_dim, latent_dim, num_snps, num_imaging_features)
# gp_layer = InformedGPLayer(num_inducing, latent_dim)
# model = InformedDKLModel(feature_extractor, gp_layer)
# likelihood = gpytorch.likelihoods.GaussianLikelihood()

# # Training loop
# def train(model, likelihood, train_loader, optimizer, mll, epochs):
#     for epoch in range(epochs):
#         model.train()
#         likelihood.train()
#         epoch_loss = 0
#         for batch_x, batch_time, batch_snp, batch_y, batch_prog in train_loader:
#             optimizer.zero_grad()
#             output, prog_prob, attention = model(batch_x, batch_time, batch_snp)
            
#             # GP regression loss
#             loss = -mll(output, batch_y)
            
#             # Binary classification loss for progressor prediction
#             prog_loss = nn.BCELoss()(prog_prob, batch_prog.float())
            
#             # Combine losses
#             total_loss = loss + prog_loss
#             total_loss.backward()
#             optimizer.step()
#             epoch_loss += total_loss.item()
#         print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}')

# # Use the model
# optimizer = torch.optim.Adam([
#     {'params': model.parameters()},
#     {'params': likelihood.parameters()},
# ], lr=0.01)

# mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

# train(model, likelihood, train_loader, optimizer, mll, epochs=100)

# # Prediction and analysis function
# def predict_and_analyze(model, x, time, snp_data):
#     model.eval()
#     with torch.no_grad():
#         output, prog_prob, attention = model(x, time, snp_data)
#         return output.mean, prog_prob, attention

# # Example usage and analysis
# test_x = torch.randn(10, input_dim)  # 10 test samples
# test_time = torch.rand(10)  # 10 time points
# test_snp = torch.randint(0, 3, (10, num_snps))  # 10 test samples, num_snps SNPs per sample
# predicted_y, predicted_prog_prob, attention_maps = predict_and_analyze(model, test_x, test_time, test_snp)

# # Analyze imaging-genomic associations
# def analyze_associations(attention_maps, prog_probs, threshold=0.5):
#     progressor_attention = attention_maps[prog_probs.squeeze() > threshold].mean(dim=0)
#     non_progressor_attention = attention_maps[prog_probs.squeeze() <= threshold].mean(dim=0)
    
#     diff_attention = progressor_attention - non_progressor_attention
    
#     top_progressor_features = torch.argsort(progressor_attention, descending=True)[:10]
#     top_non_progressor_features = torch.argsort(non_progressor_attention, descending=True)[:10]
#     most_different_features = torch.argsort(torch.abs(diff_attention), descending=True)[:10]
    
#     return {
#         'top_progressor_features': top_progressor_features.tolist(),
#         'top_non_progressor_features': top_non_progressor_features.tolist(),
#         'most_different_features': most_different_features.tolist(),
#     }

# associations = analyze_associations(attention_maps, predicted_prog_prob)
# print("Top features for progressors:", associations['top_progressor_features'])
# print("Top features for non-progressors:", associations['top_non_progressor_features'])
# print("Features with most different attention:", associations['most_different_features'])