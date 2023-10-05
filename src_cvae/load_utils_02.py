import torch

import litdata
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class ToRGBTensor:
    # converting gray scale images to 3 times the image to imitating RGB image dimesion
    def __call__(self, x):
        return T.functional.to_tensor(x).expand(3, -1, -1)



def calculate_mean_std(loader):
    mean = 0.0
    squared_mean = 0.0
    std = 0.0
    total_samples = 0.0
    
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        total_samples += batch_samples
        mean += data.mean(2).sum(0)
        squared_mean += data.pow(2).mean(2).sum(0)
    
    mean /= total_samples
    squared_mean /= total_samples
    std = (squared_mean - mean.pow(2)).sqrt()

    return mean.tolist(), std.tolist()

def load_car_data(batch_size=128):
    datapath = '/projects/ec232/data/'
    
    preliminary_transform = T.Compose([
        T.Resize((96, 128), antialias=True),
        ToRGBTensor(),  # Assuming this already converts to tensor
    ])

    preliminary_data = litdata.LITDataset(
        'CarRecs',
        datapath,
        override_extensions=[
            'jpg',
            'scores.npy'
        ]
    ).map_tuple(preliminary_transform, T.ToTensor())
    
    preliminary_loader = DataLoader(preliminary_data, shuffle=False, batch_size=batch_size)
    
    # Calculate mean and std
    in_mean, in_std = calculate_mean_std(preliminary_loader)
    print(f'Calculated mean: {in_mean}, std: {in_std}')
    
    postprocess = T.Compose([
        T.Resize((96, 128), antialias=True),
        ToRGBTensor(),
        T.Normalize(in_mean, in_std)
    ])

    final_data = litdata.LITDataset(
        'CarRecs',
        datapath,
        override_extensions=[
            'jpg',
            'scores.npy'
        ]
    ).map_tuple(postprocess, T.ToTensor())
    
    train_loader = DataLoader(final_data, shuffle=True, batch_size=batch_size)
    return train_loader



def reparameterize_gaussian(mean, std):
    """
    Inputs:
        mean : [torch.tensor] Mean vector. Shape: batch_size x z_dim.
        std  : [torch.tensor] Standard deviation vection. Shape: batch_size x z_dim.
    
    Output:
        z    : [torch.tensor] z sampled from the Normal distribution with mean and standard deviation given by the inputs. 
                              Shape: batch_size x z_dim.
    """

    # Sample epsilon from N(0,I)
    eps = torch.randn_like(std)

    # Calculate z using reparameterization trick
    z = mean + std*eps

    return z


def vae_loss(x, x_hat, mean, logvar):
    """
    Inputs:
        x       : [torch.tensor] Original sample
        x_hat   : [torch.tensor] Reproduced sample
        mean    : [torch.tensor] Mean mu of the variational posterior given sample x
        logvar  : [torch.tensor] log of the variance sigma^2 of the variational posterior given sample x
    """

    # Recontruction loss
    reproduction_loss = ((x - x_hat)**2).sum()

    # KL divergence
    KL_divergence = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # Get the total loss
    loss = reproduction_loss + KL_divergence

    return loss



def plot_and_save_generated_image(x_hat, save_path='generated_image.png'):
    """
    Plot the generated image and save it.

    Parameters:
    - x_hat (torch.Tensor): The generated image tensor.
    - save_path (str): The path where the generated image will be saved.
    """
    plt.figure(figsize=(3,3))
    
    # If your tensor is on a device other than CPU, make sure to bring it back to CPU
    if x_hat.device != 'cpu':
        x_hat = x_hat.cpu()

    plt.imshow(x_hat.permute(1,2,0).detach().numpy(), cmap=plt.get_cmap('gray'))
    plt.title("A generated image")
    plt.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
