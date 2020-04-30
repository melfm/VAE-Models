""" Training VAE """
import os
from os.path import exists, join

import numpy as np
import torch
import torch.utils.data
from benchmarking_dynamics_models.dynamics_dataloader import (
    DynamicsDataset, collate_fn, get_standardized_collate_fn)
from learning import EarlyStopping, ReduceLROnPlateau
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from vae import VAE

import hydra

cuda = torch.cuda.is_available()

torch.manual_seed(123)

device = torch.device("cuda" if cuda else "cpu")

@hydra.main(config_path='/home/melissa/Workspace/benchmarking_dynamics_models/conf/model/world_models.yaml',
            strict=False)
def main(cfg):
    epochs = cfg.epochs
    sequence_horizon = cfg.sequence_horizon
    batch_size = cfg.batch_size

    train_dataset = DynamicsDataset(cfg.dataset.train_path, horizon=sequence_horizon)
    collate_fn = get_standardized_collate_fn(train_dataset, keep_dims=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    state_dims = len(train_dataset.states[0][0])

    # load test dataset
    test_dataset = DynamicsDataset(cfg.dataset.test_path, horizon=1)
    collate_fn = get_standardized_collate_fn(test_dataset, keep_dims=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)

    input_size = state_dims

    ##########################
    # VAE Parameters
    ##########################
    latent_size = cfg.latent_size
    hidden_size = cfg.vae_hidden_size
    beta = cfg.beta

    model = VAE(input_size, latent_size, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(recon_x, x, mu, logsigma):
        """ VAE loss function """
        BCE = F.mse_loss(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())

        return BCE + beta * KLD


    def train(epoch):
        """ One training epoch """
        model.train()
        train_loss = 0
        for batch_idx, (states, _, _, _, _, _ ) in enumerate(train_loader):
            states = states.to(device).float()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(states)
            loss = loss_function(recon_batch, states, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 2 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, epoch * len(train_loader), len(train_loader.dataset),
                    100. * epoch / len(train_loader),
                    loss.item() / len(train_loader)))
        print('\n ====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader)))


    def test():
        """ One test epoch """
        model.eval()
        test_loss = 0
        console_print = True
        # console_count = 0

        preds = []
        gt = []

        with torch.no_grad():
            for state, _, _, _, _, _  in test_loader:
                states = torch.tensor(state).float().to(device)
                recon_batch, mu, logvar = model(states)
                test_loss += loss_function(recon_batch, states, mu, logvar).item()

                state_encoded = model.encoder(states)
                state_recons = model.decoder(state_encoded[0])
                state_recons = state_recons.cpu().data.numpy()
                state_np = state

                preds.append(state_recons)
                gt.append(state_np)

                # console_count += 1
                # if console_print & (console_count % 2000 == 0):
                #     print('\n State original : ', state_np, '\n')
                #     print('\n State reconstruct: ', state_recons, '\n')

        test_loss /= len(test_dataset)
        print('\n ====> Test set loss: {:.4f}'.format(test_loss))

        preds = np.stack(preds)
        gt = np.stack(gt)
        error = (preds - gt)**2
        if console_print:
            print("Mean Error: {}".format(error.mean(0)[0]))
            print("Min  Error: {}".format(error.min(0)[0]))
            print("Max  Error: {}".format(error.max(0)[0]))

        return test_loss

    def save_checkpoint(state, is_best, filename, best_filename):
        """ Save state in filename. Also save in best_filename if is_best. """
        torch.save(state, filename)
        if is_best:
            torch.save(state, best_filename)
            
    # check vae dir exists, if not, create it
    vae_dir = join(cfg.logdir, 'vae')
    if not exists(vae_dir):
        os.makedirs(vae_dir)
        os.makedirs(join(vae_dir, 'samples'))

    reload_file = join(vae_dir, 'best.tar')
    if not cfg.noreload and exists(reload_file):
        state = torch.load(reload_file)
        print("Reloading model at epoch {}"
            ", with test error {}".format(
                state['epoch'],
                state['precision']))
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        earlystopping.load_state_dict(state['earlystopping'])


    cur_best = None

    for epoch in range(1, epochs + 1):
        train(epoch)
        test_loss = test()
        scheduler.step(test_loss)
        earlystopping.step(test_loss)

        # checkpointing
        best_filename = join(vae_dir, 'best.tar')
        filename = join(vae_dir, 'checkpoint.tar')
        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'precision': test_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict()
        }, is_best, filename, best_filename)

        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(epoch))
            break

if __name__ == "__main__":
    main()