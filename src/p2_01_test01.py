import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set the main device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose the path where you will/have already saved the MNIST dataset
data_dir = '/fp/projects01/ec232/data'

# Get dataset
transform = transforms.ToTensor()
train_set = MNIST(root=data_dir, train=True, download=True, transform=transform)
N = len(train_set)

# Set a suitable batch size
batch_size = 128

# Create data loader
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)


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

class Encoder(nn.Module):
    """ Convolutional encoder for the VAE. """

    def __init__(self, z_dim, n_channels):
        super().__init__()

        feature_dim = 32 * 6 * 6
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.mean_fc = nn.Linear(feature_dim, z_dim)
        self.logvar_fc = nn.Linear(feature_dim, z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)
        return mean, logvar

class Decoder(nn.Module):
    """ Convolutional decoder for the VAE. """

    def __init__(self, z_dim, n_channels):
        super().__init__()
        
        feature_dim = 32 * 6 * 6
        self.fc = nn.Linear(z_dim, feature_dim)
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)
        self.conv1 = nn.ConvTranspose2d(16, n_channels, kernel_size=3, stride=2, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 32, 6, 6)
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))
        return x

class VAE(nn.Module):
    """ Variational Auto-Encoder class. """

    def __init__(self, z_dim, n_channels=1):
        """
        Inputs:
            z_dim       : [int] Dimension of the latent variable z
            n_channels  : Number of channels in the input image.
        """
        super().__init__()
        self.z_dim = z_dim
        self.n_channels = n_channels

        self.encoder = Encoder(z_dim, n_channels)
        self.decoder = Decoder(z_dim, n_channels)

    def forward(self, x):

        # Get the mean and log variance from the encoder
        mean, logvar = self.encoder(x)

        # Calculate the standard deviation. Note: in log-space, squareroot is divide by two
        std = torch.exp(logvar / 2) 

        # Sample the latent using the reparameterization trick
        z = reparameterize_gaussian(mean, std)

        # Get reconstructed x from the decoder
        x_hat = self.decoder(z)

        return x_hat, mean, logvar
    
    def get_latents(self, x):
        """
        Function that returns the latents z given input x. 
        Useful for evaluating the latent space Z after training.
        """
        
        # Get the mean and log variance from the encoder
        mean, logvar = self.encoder(x)

        # Calculate the standard deviation. Note: in log-space, squareroot is divide by two
        std = torch.exp(logvar / 2)

        # Sample the latent using the reparameterization trick
        z = reparameterize_gaussian(mean, std)
        
        return z
    
# Choose the latent dimension. You can for example use z_dim=2
z_dim = 2

# Initialize the VAE
model = VAE(z_dim, n_channels=1).to(device)

# Choose the training parameters. Feel free to change them.
epochs = 10
lr = 0.01

# Initialize the optimizer
optimizer = Adam(model.parameters(), lr = lr)

# Train for a few epochs
model.train()
for epoch in range(epochs):
    train_bar = tqdm(iterable=train_loader)
    for i, (x, c) in enumerate(train_bar):
        
        x = x.to(device)
        # Get x_hat, mean, and logvar from the VAE model
        x_hat, mean, logvar = model(x)

        # Get vae loss
        loss = vae_loss(x, x_hat, mean, logvar)

        # Update model parameters based on loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_bar.set_description(f'Epoch [{epoch+1}/{epochs}]')
        train_bar.set_postfix(loss = loss.item() / len(x))

# Generate new images from randomly sampled latents z

model.eval()
with torch.no_grad():
    # Sample a random latent z from the prior N(0,I)
    z = torch.randn(1, z_dim).to(device)

    # Generate a new image from z using the decoder
    x_hat = model.decoder(z)

    # Squeeze batch dimension and move image to cpu
    x_hat = x_hat.squeeze(0).cpu().detach()


# Plot the generated image
plt.figure(figsize=(3,3))
plt.imshow(x_hat.permute(1,2,0), cmap=plt.get_cmap('gray'))
plt.title("A generated image")
plt.axis('off')
# plt.show()
# Save the figure
plt.savefig('generated_image.png', bbox_inches='tight', pad_inches=0)
# close the figure after saving it
plt.close()


