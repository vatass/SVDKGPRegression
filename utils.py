import pickle
import torch 

## Attention: This Function needs to be customized because now everything is hard coded!! TODO
def transfer_weights(model):
    # Load the stored weights and biases
    with open('best_mlp_weights.pkl', 'rb') as f:
        params = pickle.load(f)

    coefs = params['coefs']
    intercepts = params['intercepts']
    
    for i in range(len(coefs)):
        print(f'Layer {i + 1} weights shape: {coefs[i].shape}')
        print(f'Layer {i + 1} biases shape: {intercepts[i].shape}')

    ## Now get the weights of the pytorch model to make sure of the shape they have
    for name, param in model.named_parameters():
        print(name, param.shape)

    # Transfer the weights and biases to the PyTorch model
    model.fc1.weight.data = torch.tensor(coefs[0].T)  # Transpose to match PyTorch's weight shape (That's Verified!)
    model.fc1.bias.data = torch.tensor(intercepts[0])

    model.fc2.weight.data = torch.tensor(coefs[1].T)
    model.fc2.bias.data = torch.tensor(intercepts[1])

    model.fc3.weight.data = torch.tensor(coefs[2].T)
    model.fc3.bias.data = torch.tensor(intercepts[2])

    model.fc4.weight.data = torch.tensor(coefs[3].T)
    model.fc4.bias.data = torch.tensor(intercepts[3])

    model.fc5.weight.data = torch.tensor(coefs[4].T)
    model.fc5.bias.data = torch.tensor(intercepts[4])

    # The output layer should be tuned by the GP 
    # model.output.weight.data = torch.tensor(coefs[5].T)
    # model.output.bias.data = torch.tensor(intercepts[5])

    return model

def freeze_layers(model, num_layers_to_train=1):
    """Freeze all layers except the last num_layers_to_train layers."""
    for param in list(model.feature_extractor.parameters())[:-num_layers_to_train]:
        param.requires_grad = False
