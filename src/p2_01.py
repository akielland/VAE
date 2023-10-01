import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from tqdm import tqdm

from load_utils import load_car_data, plot_and_save_generated_image, vae_loss, reparameterize_gaussian

# Set the main device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the transformed car dataset
train_loader = load_car_data(batch_size=128)

# Do I need this:
N = len(train_loader.dataset)


class Encoder(nn.Module):
    def __init__(self, z_dim, n_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=2)  # 96x128 -> 47x63 (no padding)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2) # 47x63 -> 23x31 (no padding)
        feature_dim = 32 * 23 * 31 # calculated based on input dimensions and conv layers
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
    def __init__(self, z_dim, n_channels):
        super().__init__()
        
        feature_dim = 32 * 23 * 31
        self.fc = nn.Linear(z_dim, feature_dim)
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)
        self.conv1 = nn.ConvTranspose2d(16, n_channels, kernel_size=3, stride=2, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        # x = x.view(-1, 32, 6, 6)
        x = x.view(-1, 32, 23, 31) # reshape to match the expected dimensions for deconv layers
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
n_channels=3

# Initialize the VAE
model = VAE(z_dim, n_channels).to(device)

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

    plot_and_save_generated_image(x_hat)



