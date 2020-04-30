"""
Variational encoder model, used as a dynamics model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, hidden_size, latent_size, output_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size

        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        reconstruction = self.fc3(x)
        return reconstruction

class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logsigma = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma

class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, input_size, latent_size, hidden_size=50):
        super(VAE, self).__init__()

        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(hidden_size, latent_size, input_size)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma