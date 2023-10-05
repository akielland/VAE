import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam, AdamW
from tqdm import tqdm

from load_utils_02 import load_car_data, plot_and_save_generated_image, vae_loss, reparameterize_gaussian

# Set the main device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the transformed car dataset
train_loader = load_car_data(batch_size=128)


class CEncoder(nn.Module):
    def __init__(self, n_channels, n_classes, z_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(n_channels + 1, 16, kernel_size=3, stride=2)  # 96x128 -> 47x63 (no padding)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2) # 47x63 -> 23x31 (no padding)
        self.conv3 = nn.Conv2d(32, 128, kernel_size = 3, stride = 2) # 23x31 -> 11X15 (no padding)

        # feature_dim = 11 * 15 * 128 
        # calculated based on input dimensions and conv layers
        # Dummy forward pass to calculate feature_dim
        x_dummy = torch.zeros(1, n_channels + 1, 96, 128)
        self.feature_dim = self._get_feature_dim(x_dummy)

        # Linear layers
        self.mean_fc = nn.Linear(self.feature_dim, z_dim)
        self.logvar_fc = nn.Linear(self.feature_dim, z_dim)
        # token representing th Moira rating estimated as a one-hot encoding, but could hve been one integer I suppose. Why we learn it...
        self.cls_token = nn.Linear(self.feature_dim, n_classes) 

    def _get_feature_dim(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.flatten(1)
        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)
        cls_token = self.cls_token(x)
        
        return mean, logvar, cls_token

class CDecoder(nn.Module):
    def __init__(self, n_channels, n_classes, z_dim, feature_dim):
        super().__init__()
        
        # feature_dim = 128 * 11 * 15
        self.feature_dim = feature_dim
        self.fc = nn.Linear(z_dim + n_classes, self.feature_dim)

        self.conv1 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2)
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)
        self.conv3 = nn.ConvTranspose2d(16, n_channels, kernel_size=3, stride=2, output_padding=1)

    def forward(self, z_cls_token):
        x = F.relu(self.fc(z_cls_token))
        x = x.view(-1, 128, 11, 15) # reshape to match the expected dimensions for deconv layers
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_hat = torch.sigmoid(self.conv3(x))
        
        return x_hat

class CVAE(nn.Module):
    def __init__(self, z_dim, n_classes=10, n_channels=3, img_size=[96,128]):
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.img_size = img_size

        self.encoder = CEncoder(n_channels, n_classes, z_dim)
        feature_dim = self.encoder.feature_dim
        self.decoder = CDecoder(n_channels, n_classes, z_dim, feature_dim)

        # Add learnable class token. Isn't these embeddings of the token classes?
        self.cls_param = nn.Parameter(torch.zeros(n_classes, *img_size))

    def get_cls_emb(self, c):
        return self.cls_param[c].unsqueeze(1)  # c picks row for a given class/rating

    def forward(self, x, c):
        """
        Args:
            x   : [torch.Tensor] Image input of shape [batch_size, n_channels, *img_size]
            c   : [torch.Tensor] Class labels (here=Moira rating) for x of shape [batch_size], where class in indicated by a
        """

        assert x.shape[1:] == (self.n_channels, *self.img_size), f'Expected input x of shape [batch_size, {[self.n_channels, *self.img_size]}], but got {x.shape}'
        assert c.shape[0] == x.shape[0], f'Inputs x and c must have same batch size, but got {x.shape[0]} and {c.shape[0]}'
        assert len(c.shape) == 1, f'Input c should have shape [batch_size], but got {c.shape}'

        # Get cls embedding
        cls_emb = self.get_cls_emb(c)

        # Concatenate cls embedding to the input
        x = torch.cat((x, cls_emb), dim=1)

        # Get the mean, logvar, and cls token from the encoder
        mean, logvar, cls_token = self.encoder(x)

        # Calculate the standard deviation. Note: in log-space, squareroot is divide by two
        std = torch.exp(logvar / 2)

        # Sample the latent using the reparameterization trick
        z = reparameterize_gaussian(mean, std)

        # Concatenate cls token to z
        z = torch.cat((z, F.softmax(cls_token, dim=1)), dim=1)

        # Get reconstructed x from the decoder
        x_hat = self.decoder(z)
        
        return x_hat, mean, logvar, cls_token
    


# Choose the latent dimension. You can for example use z_dim=2
z_dim = 200

# Initialize the CVAE model
model = CVAE(z_dim, n_classes=10, n_channels=3, img_size=[96,128]).to(device)

# Choose the training parameters. Feel free to change them.
epochs = 20
lr = 0.001
weight_decay = 0.1
cls_loss_weight = 10

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = Adam(model.parameters(), lr = lr)


# Train over epochs
model.train()
loss_train = []
for epoch in range(epochs):
    loss_epoch = 0
    
    
    train_bar = tqdm(iterable=train_loader)   # train_bar: the train_loader, wrapped with a progress bar
    for i, (x, c) in enumerate(train_bar):
        x = x.to(device)
        c = c.to(device)
        
        # Get rating for Moira 
        moira_ratings_as_class = c.squeeze(0, 1, 2)[:, 1]   # one number for each image in the batch
        
        # Get x_hat, mean, logvar, and cls_token from the conditioned_model
        x_hat, mean, logvar, cls_token = model(x, moira_ratings_as_class)

        # Get vae loss 
        vae_loss_batch = vae_loss(x, x_hat, mean, logvar)
        # print(cls_token.shape)
       
        # Get cross entropy loss for the cls token
        cls_loss_batch = F.cross_entropy(cls_token, F.one_hot(moira_ratings_as_class, num_classes=10).double(), reduction='sum')

        # Add the losses as a weighted sum. NB: We weight the cls_loss by 10 here, but feel free to tweak it.
        loss = vae_loss_batch + cls_loss_batch * cls_loss_weight
        loss_epoch += loss.item() / len(x)

        # Update model parameters based on loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_bar.set_description(f'Epoch [{epoch+1}/{epochs}]')
        train_bar.set_postfix(loss = loss.item() / len(x))


    loss_train.append(loss_epoch / len(train_loader))

# Save the model and loss per epoch
path = "../output_cvae/CVAE_model_01.pt"
state_dict = {
    "model": model.state_dict(),  # Save model parameters
    "loss": loss_train  # Save the training loss
}

# Save the model
try:
    torch.save(state_dict, path)
    print("Model saved!")
except Exception as e:
    print(f"Could not save model. Error: {e}")


# Generate new images from randomly sampled latents z

rating = 9

model.eval()
with torch.no_grad():
    # Sample a random latent z from the prior N(0,I)
    z = torch.randn(1, z_dim).to(device)

    one_hot_rating = F.one_hot(torch.tensor(rating).unsqueeze(0), num_classes=10).to(device) 
    z = torch.cat((z, one_hot_rating), dim=1)
    # Generate a new image from z using the decoder
    x_hat = model.decoder(z)

    # Squeeze batch dimension and move image to cpu
    x_hat = x_hat.squeeze(0).cpu().detach()

    plot_and_save_generated_image(x_hat)



