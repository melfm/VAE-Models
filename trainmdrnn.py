""" Recurrent model training """
from functools import partial
from os import mkdir
from os.path import exists, join
import pickle

import numpy as np
import torch
import torch.nn.functional as f
from benchmarking_dynamics_models.dynamics_dataloader import (
    DynamicsDataset, collate_fn, get_standardized_collate_fn)
from mdrnn import MDRNN,MDRNNCell, gmm_loss
from torch.utils.data import DataLoader
import torch.distributions as D

from learning import EarlyStopping
from learning import ReduceLROnPlateau
from vae import VAE

import hydra

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(config_path='/home/melissa/Workspace/benchmarking_dynamics_models/conf/model/world_models.yaml',
            strict=False)
def main(cfg):
    ################
    # Constants
    ################

    epochs = cfg.epochs
    sequence_horizon = cfg.sequence_horizon
    batch_size = cfg.batch_size

    rnn_hidden_size = cfg.rnn_hidden_size
    latent_size = cfg.latent_size
    vae_hidden_size = cfg.vae_hidden_size
    num_gaussians = cfg.num_gaussians

    ##################
    # Loading datasets
    ##################
    # Data Loading
    dataset = DynamicsDataset(cfg.dataset.train_path, horizon=sequence_horizon)
    collate_fn = get_standardized_collate_fn(dataset, keep_dims=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    action_dims = len(dataset.actions[0][0])
    state_dims = len(dataset.states[0][0])

    # load test dataset
    val_dataset = DynamicsDataset(cfg.dataset.test_path, horizon=sequence_horizon)
    collate_fn = get_standardized_collate_fn(val_dataset, keep_dims=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)

    # Only do 1-step predictions
    test_dataset = DynamicsDataset(cfg.dataset.test_path, horizon=1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    input_size = state_dims
    action_size = action_dims

    # Loading VAE
    vae_file = join(cfg.logdir, 'vae', 'best.tar')
    assert exists(vae_file), "No trained VAE in the logdir..."
    state = torch.load(vae_file)
    print("Loading VAE at epoch {} "
        "with test error {}".format(
            state['epoch'], state['precision']))

    vae = VAE(input_size, latent_size, vae_hidden_size).to(device)
    vae.load_state_dict(state['state_dict'])

    # Loading model
    rnn_dir = join(cfg.logdir, 'mdrnn')
    rnn_file = join(rnn_dir, 'best.tar')

    if not exists(rnn_dir):
        mkdir(rnn_dir)

    mdrnn = MDRNN(latent_size, action_size, rnn_hidden_size, num_gaussians)

    mdrnn.to(device)
    optimizer = torch.optim.Adam(mdrnn.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30)

    if exists(rnn_file) and not cfg.noreload:
        rnn_state = torch.load(rnn_file)
        print("Loading MDRNN at epoch {} "
            "with test error {}".format(
                rnn_state["epoch"], rnn_state["precision"]))
        mdrnn.load_state_dict(rnn_state["state_dict"])
        optimizer.load_state_dict(rnn_state["optimizer"])
        scheduler.load_state_dict(state['scheduler'])
        earlystopping.load_state_dict(state['earlystopping'])

    def save_checkpoint(state, is_best, filename, best_filename):
        """ Save state in filename. Also save in best_filename if is_best. """
        torch.save(state, filename)
        if is_best:
            torch.save(state, best_filename)

    def to_latent(obs, next_obs, batch_size=1, sequence_horizon=1):
        """ Transform observations to latent space.

        :args obs: 5D torch tensor (batch_size, SEQ_LEN, ASIZE, SIZE, SIZE)
        :args next_obs: 5D torch tenbayersor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

        :returns: (latent_obs, latent_next_obs)
            - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
            - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        """
        with torch.no_grad():

            # 1: is to ignore the reconstruction, just get mu and logsigma
            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                vae(x)[1:] for x in (obs, next_obs)]
            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(batch_size,
                    sequence_horizon, latent_size)
                for x_mu, x_logsigma in
                [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
        return latent_obs, latent_next_obs

    def get_loss(latent_obs, action, reward, terminal,
                latent_next_obs, include_reward: bool):
        """ Compute losses.

        The loss that is computed is:
        (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
            BCE(terminal, logit_terminal)) / (LSIZE + 2)
        The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
        approximately linearily with LSIZE. All losses are averaged both on the
        batch and the sequence dimensions (the two first dimensions).

        :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args reward: (BSIZE, SEQ_LEN) torch tensor
        :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """
        latent_obs, action,\
            reward, terminal,\
            latent_next_obs = [arr.transpose(1, 0)
                            for arr in [latent_obs, action,
                                        reward, terminal,
                                        latent_next_obs]]

        mus, sigmas, logpi,pi, rs, ds = mdrnn(action, latent_obs)
        gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
        # bce = f.binary_cross_entropy_with_logits(ds, terminal)
        bce = 0
        if include_reward:
            mse = f.mse_loss(rs, reward)
            scale = latent_size + 2
        else:
            mse = 0
            scale = latent_size + 1
        # loss = (gmm + bce + mse) / scale
        loss = (gmm + mse) / scale
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


    def data_pass(epoch, train, include_reward):
        """ One pass through the data """
        if train:
            mdrnn.train()
            loader = train_loader
        else:
            mdrnn.eval()
            loader = val_loader

        cum_loss = 0
        cum_gmm = 0
        cum_bce = 0
        cum_mse = 0

        for batch_index, (states, actions, next_states, rewards, _, _ )in enumerate(loader):
            #import pdb;pdb.set_trace()
            if batch_index > 1000:
                break
            states = states.to(device)
            next_states = next_states.to(device)
            rewards = rewards.to(device)
            actions = actions.to(device)
            # Not sure why terminals matter here but we dont store them in the dataset
            # so just set them all to false.
            terminal = torch.zeros(batch_size, sequence_horizon).to(device)

            latent_obs, latent_next_obs = to_latent(states, next_states,
                batch_size=batch_size, sequence_horizon=sequence_horizon)
        
            if train:
                losses = get_loss(latent_obs, actions, rewards,
                                terminal, latent_next_obs, include_reward)

                optimizer.zero_grad()
                losses['loss'].backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    losses = get_loss(latent_obs, actions, rewards,
                                    terminal, latent_next_obs,include_reward)

            cum_loss += losses['loss'].item()
            cum_gmm += losses['gmm'].item()
            # cum_bce += losses['bce'].item()
            cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
                losses['mse']
            data_size = len(loader)

            if batch_index % 100 == 0:
                print("Train" if train else "Test")

                print("loss={loss:10.6f} bce={bce:10.6f} "
                                    "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                                        loss=cum_loss / data_size, bce=cum_bce / data_size,
                                        gmm=cum_gmm / latent_size / data_size, mse=cum_mse / data_size))

        return cum_loss * batch_size / len(loader.dataset)

    
    def predict():
        mdrnn.eval()

        preds = []
        gt = []
        n_episodes = test_dataset[-1][-2] + 1
        predictions = [[] for _ in range(n_episodes)]
        with torch.no_grad():
            for batch_index, (states, actions, next_states, rewards, episode, timesteps) in enumerate(test_loader):

                states = states.to(device)
                next_states = next_states.to(device)
                rewards = rewards.to(device)
                actions = actions.to(device)

                latent_obs, _ = to_latent(states,
                    next_states, batch_size=1,sequence_horizon=1)

                # Check model's next state predictions
                mus, sigmas, logpi, _ , _, _ = mdrnn(actions, latent_obs)
                mix = D.Categorical(logpi)
                comp = D.Independent(D.Normal(mus, sigmas), 1)
                gmm = D.MixtureSameFamily(mix, comp)
                sample = gmm.sample()

                decoded_states = vae.decoder(sample).squeeze(0)
                decoded_states = decoded_states.cpu().detach().numpy()
                preds.append(decoded_states)

                for i in range(len(states)):
                    predictions[episode[i].int()].append(np.expand_dims(decoded_states[i], axis=0))


                gt.append(next_states.cpu().detach().numpy())
            #import pdb;pdb.set_trace()
            predictions = [np.stack(p) for p in predictions]
            preds = np.asarray(preds)
            gt = np.asarray(gt).squeeze(1)
            error = (preds - gt)**2

        path = cfg.logdir + '/' + cfg.resname + '.pkl'
        pickle.dump(predictions, open(path, 'wb'))

        print("Mean Error: {}".format(error.mean(0)[0]))
        print("Min  Error: {}".format(error.min(0)[0]))
        print("Max  Error: {}".format(error.max(0)[0]))

    train = partial(data_pass, train=True, include_reward=cfg.include_reward)
    test = partial(data_pass, train=False, include_reward=cfg.include_reward)

    cur_best = None
    for e in range(epochs):
        train(e)
        test_loss = test(e)
        predict()

        scheduler.step(test_loss)
        #earlystopping.step(test_loss)

        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss
        checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
        save_checkpoint({
            "state_dict": mdrnn.state_dict(),
            "optimizer": optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict(),
            "precision": test_loss,
            "epoch": e}, is_best, checkpoint_fname,
                        rnn_file)

        #if earlystopping.stop:
        #    print("End of Training because of early stopping at epoch {}".format(e))
        #    break

if __name__ == "__main__":
    main()