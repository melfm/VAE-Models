import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from benchmarking_dynamics_models.dynamics_dataloader import (
    DynamicsDataset, collate_fn, get_standardized_collate_fn)
from torch.utils.data import DataLoader


class DBlock(nn.Module):
    """ A basic building block to parameterize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma

class Decoder(nn.Module):
    """ The decoder layer converting state to observation.
    The output of this layer are probabilities of
    elements being 1.
    """
    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p

class TD_VAE(nn.Module):
    """The full TD_VAE model with jumpy predictions.
    """

    def __init__(self,
                 x_size,
                 b_size,
                 z_size,
                 hidden_size=50,
                 decoder_hidden_size=200):
        super(TD_VAE, self).__init__()
        self.x_size = x_size
        self.b_size = b_size
        self.z_size = z_size
        # Input to LSTM (sequence_len, batch_size, input_size)
        self.lstm = nn.LSTM(input_size=self.x_size,
                            hidden_size=self.b_size,
                            batch_first=True)
        # Layer 2
        self.l2_b_to_z = DBlock(b_size, hidden_size, z_size)
        # Layer 1
        self.l1_b_to_z = DBlock(b_size + z_size,
                                hidden_size,
                                z_size)

        # Inference layers
        # Layer 2
        self.l2_infer_z = DBlock(b_size + 2*z_size + 1,
                                 hidden_size, z_size)
        # Layer 1
        self.l1_infer_z = DBlock(b_size + 2*z_size + z_size + 1,
                                 hidden_size, z_size)

        # Given the state at time t1, model state at t2
        # through state transition
        self.l2_transition_z = DBlock(2*z_size + 1,
                                      hidden_size,
                                      z_size)
        self.l1_transition_z = DBlock(2*z_size + z_size + 1,
                                      hidden_size,
                                      z_size)

        # State -> observation
        self.z_to_x = Decoder(2*z_size,
                              decoder_hidden_size,
                              x_size)

    def forward(self, input):
        self.batch_size = input.size()[0]
        self.x = input
        # Aggregate the belief b
        self.b, (_, _) = self.lstm(self.x)

    def loss_function(self, t1, t2, beta, device):
        """ Calculate the jumpy VD-VAE loss, which is corresponding to
        the equation (6) and equation (8) in the reference.
        """
        # These are heirarchical layers, where the latent
        # variables in the higher layer are sampled first
        # and their results are passed to the lower layer.

        # sample a state at time t2 (see the reparametralization
        # trick is used)
        # z in layer 2

        t2_l2_z_mu, t2_l2_z_logsigma = self.l2_b_to_z(self.b[t2,:,:])
        t2_l2_z_epsilon = torch.randn_like(t2_l2_z_mu)
        t2_l2_z = t2_l2_z_mu + torch.exp(t2_l2_z_logsigma)*t2_l2_z_epsilon

        # z in layer 1
        t2_l1_z_mu, t2_l1_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[t2,:,:], t2_l2_z),dim = -1))
        t2_l1_z_epsilon = torch.randn_like(t2_l1_z_mu)
        t2_l1_z = t2_l1_z_mu + torch.exp(t2_l1_z_logsigma)*t2_l1_z_epsilon

        # concatenate z from layer 1 and layer 2
        t2_z = torch.cat((t2_l1_z, t2_l2_z), dim = -1)

        # sample a state at time t1
        # infer state at time t1 based on states at time t2
        delta_t = torch.tensor(t2-t1).repeat(self.b.shape[1]).\
            to(device).float().unsqueeze(1)

        t1_l2_qs_z_mu, t1_l2_qs_z_logsigma = self.l2_infer_z(
            torch.cat((self.b[t1,:,:], t2_z, delta_t), dim = -1))

        t1_l2_qs_z_epsilon = torch.randn_like(t1_l2_qs_z_mu)
        t1_l2_qs_z = t1_l2_qs_z_mu + \
            torch.exp(t1_l2_qs_z_logsigma)*t1_l2_qs_z_epsilon

        t1_l1_qs_z_mu, t1_l1_qs_z_logsigma = self.l1_infer_z(
            torch.cat((self.b[t1,:,:], t2_z, t1_l2_qs_z, delta_t), dim = -1))
        t1_l1_qs_z_epsilon = torch.randn_like(t1_l1_qs_z_mu)
        t1_l1_qs_z = t1_l1_qs_z_mu + torch.exp(t1_l1_qs_z_logsigma)*t1_l1_qs_z_epsilon

        t1_qs_z = torch.cat((t1_l1_qs_z, t1_l2_qs_z, delta_t), dim = -1)

        # After sampling states z from the variational distribution,
        # we can calculate the loss.

        # state distribution at time t1 based on belief at time 1
        t1_l2_pb_z_mu, t1_l2_pb_z_logsigma = self.l2_b_to_z(self.b[t1,:,:])
        t1_l1_pb_z_mu, t1_l1_pb_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[t1,:,:], t1_l2_qs_z),dim = -1))

        # state distribution at time t2 based on states at time t1
        # and state transition
        t2_l2_t_z_mu, t2_l2_t_z_logsigma = self.l2_transition_z(t1_qs_z)
        t2_l1_t_z_mu, t2_l1_t_z_logsigma = self.l1_transition_z(
            torch.cat((t1_qs_z, t2_l2_z), dim = -1))

        # observation distribution at time t2 based on state at time t2
        t2_x_prob = self.z_to_x(t2_z)

        # Losses

        # KL divergence between z distribution at time t1 based on
        # variational distribution
        # (inference model) and z distribution at time t1 based on belief.
        # This divergence is between two normal distributions and
        # it can be calculated analytically

        # KL divergence between t1_l2_pb_z, and t1_l2_qs_z
        loss = 0.5*torch.sum(((t1_l2_pb_z_mu - \
                             t1_l2_qs_z)/torch.exp(t1_l2_pb_z_logsigma))**2,-1) + \
               torch.sum(t1_l2_pb_z_logsigma, -1) - \
                   torch.sum(t1_l2_qs_z_logsigma, -1)

        # KL divergence between t1_l1_pb_z and t1_l1_qs_z
        loss += 0.5*torch.sum(((t1_l1_pb_z_mu - \
            t1_l1_qs_z)/torch.exp(t1_l1_pb_z_logsigma))**2,-1) + \
               torch.sum(t1_l1_pb_z_logsigma, -1) - \
                   torch.sum(t1_l1_qs_z_logsigma, -1)
        # Hyperparameter for the KL term
        loss *= beta

        # The following four terms estimate the KL divergence
        # between the z distribution at time t2
        # based on variational distribution (inference model)
        # and z distribution at time t2 based on transition.
        # In contrast with the above KL divergence for z
        # distribution at time t1, this KL divergence
        # can not be calculated analytically because the
        # transition distribution depends on z_t1, which
        # is sampled after z_t2. Therefore, the KL divergence
        # is estimated using samples

        # state log probabilty at time t2 based on belief
        loss += torch.sum(-0.5*t2_l2_z_epsilon**2 - \
                         0.5*t2_l2_z_epsilon.new_tensor(2*np.pi) - \
                         t2_l2_z_logsigma, dim = -1)
        loss += torch.sum(-0.5*t2_l1_z_epsilon**2 - \
                         0.5*t2_l1_z_epsilon.new_tensor(2*np.pi) - \
                         t2_l1_z_logsigma, dim = -1)

        # state log probabilty at time t2 based on transition
        loss += torch.sum(0.5*((t2_l2_z - t2_l2_t_z_mu)/torch.exp(t2_l2_t_z_logsigma))**2 +\
             0.5*t2_l2_z.new_tensor(2*np.pi) + t2_l2_t_z_logsigma, -1)
        loss += torch.sum(0.5*((t2_l1_z - t2_l1_t_z_mu)/torch.exp(t2_l1_t_z_logsigma))**2 + \
            0.5*t2_l1_z.new_tensor(2*np.pi) + t2_l1_t_z_logsigma, -1)

        loss += -torch.sum(self.x[t2,:,:]*torch.log(t2_x_prob) + \
            (1-self.x[t2,:,:])*torch.log(1-t2_x_prob), -1)
        loss = torch.mean(loss)

        return loss

    def rollout(self, data, t1, t2, device):
        self.forward(data)

        # At time t1-1, we sample a state z based on belief at time t1-1
        l2_z_mu, l2_z_logsigma = self.l2_b_to_z(self.b[t1-1,:,:])
        l2_z_epsilon = torch.randn_like(l2_z_mu)
        l2_z = l2_z_mu + torch.exp(l2_z_logsigma)*l2_z_epsilon

        l1_z_mu, l1_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[t1-1,:,:], l2_z), dim = -1))
        l1_z_epsilon = torch.randn_like(l1_z_mu)
        l1_z = l1_z_mu + torch.exp(l1_z_logsigma)*l1_z_epsilon

        delta_t = torch.tensor(t2-t1).repeat(self.b.shape[1]).\
            to(device).float().unsqueeze(1)

        current_z = torch.cat((l1_z, l2_z, delta_t), dim = -1)

        rollout_x = []

        for _ in range(t2 - t1 + 1):
            # predicting states after time t1 using state transition
            next_l2_z_mu, next_l2_z_logsigma = self.l2_transition_z(current_z)
            next_l2_z_epsilon = torch.randn_like(next_l2_z_mu)
            next_l2_z = next_l2_z_mu + \
                torch.exp(next_l2_z_logsigma)*next_l2_z_epsilon

            next_l1_z_mu, next_l1_z_logsigma  = self.l1_transition_z(
                torch.cat((current_z, next_l2_z), dim = -1))
            next_l1_z_epsilon = torch.randn_like(next_l1_z_mu)
            next_l1_z = next_l1_z_mu + \
                torch.exp(next_l1_z_logsigma)*next_l1_z_epsilon

            next_z = torch.cat((next_l1_z, next_l2_z), dim = -1)

            # generate an observation x_t1 at time t1 based on
            # sampled state z_t1 - decoder extracts the first
            # component from the z of the first layer
            next_x = self.z_to_x(next_z)
            rollout_x.append(next_x)
            current_z = next_z
            current_z = torch.cat((current_z, delta_t), dim = -1)

        rollout_x = torch.stack(rollout_x, dim = 1)

        return rollout_x


#########################################################################
## Training & Evaluation of TD-VAE on Half-Cheetah
#########################################################################

parser = argparse.ArgumentParser(description='TD-VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--beta', type=float, default=5e-5,
                    help='KL beta param')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

if __name__=="__main__":
    #################
    # Dataset Loader
    #################
    sequence_horizon = 1
    batch_size = 100

    # dataset = DynamicsDataset('dyna_benchmarking/HalfCheetah/experience.pth',
    #                           horizon=sequence_horizon)
    dataset = DynamicsDataset('dyna_benchmarking/benchmarking_dynmodels/train/Cartpole/2020_03_13_09_43_14.377385/experience.pth.tar',
                               horizon=sequence_horizon)
    collate_fn = get_standardized_collate_fn(dataset, keep_dims=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    action_dims = len(dataset.actions[0][0])
    state_dims = len(dataset.states[0][0])

    # load test dataset
    test_dataset = DynamicsDataset('dyna_benchmarking/benchmarking_dynmodels/test/Cartpole/2019_10_04_08_08_08.703338/experience.pth.tar',
                                    horizon=sequence_horizon)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  drop_last=True, collate_fn=collate_fn)

    ######################
    # Build a TD-VAE model
    ######################
    input_size = state_dims
    belief_state_size = 50
    state_size = 8
    td_vae_model= TD_VAE(input_size,
                         belief_state_size,
                         state_size)
    td_vae_model = td_vae_model.cuda()

    optimizer = optim.Adam(td_vae_model.parameters(), lr = 0.0005)
    beta = args.beta

    print(td_vae_model)

    # train model
    n_epochs = 50
    t1 = 60
    state_jump = 20

    td_vae_model.train()

    for epoch in tqdm.tqdm(range(n_epochs), desc="Epoch: "):
        pbar = tqdm.tqdm(train_loader, leave=epoch==n_epochs)
        for states, actions, next_states, rewards, episodes, timesteps in pbar:
            # normalize the obs
            states = (states - \
                dataset.stats['states_min'][0])/(dataset.stats['states_max'][0]- \
                    dataset.stats['states_min'][0])

            optimizer.zero_grad()
            states = states.cuda()
            td_vae_model.forward(states)

            t_1 = np.random.choice(t1)
            t_2 = t_1 + np.random.choice(np.arange(1, state_jump))
            loss = td_vae_model.loss_function(t_1, t_2, beta, device)

            loss.backward()
            optimizer.step()
            pbar.set_description("Epoch: {:>4d}, loss: {:.2f}".format(
                epoch, loss.item()))

    td_vae_model.eval()

    preds = []
    gt = []
    console_print = 0
    with torch.no_grad():
        for states, actions, next_states, rewards, episodes, timesteps in tqdm.tqdm(test_loader):
            # Slide through the traj and predict 10-step rollouts
            # (t1 = 0, t2 = 10), (t1=1, t2=11) ...
            # Q : Does the state need to be reset ?
            # Be careful with this indexing, as inside rollout, its going to sample belief from t1-1
            # but we are actually predicting t1 onwards.
            t1 = 1
            t2 = 10
            traj_len = states.shape[0]
            while(t2 < 100-1):
                states = states.cpu()
                states = (states - \
                    dataset.stats['states_min'][0])/(dataset.stats['states_max'][0]- \
                    dataset.stats['states_min'][0])

                next_states = (next_states - \
                    dataset.stats['states_min'][0])/(dataset.stats['states_max'][0]- \
                        dataset.stats['states_min'][0])
                states = states.cuda()
                # This gives predictions for 10-steps, from t1 ... t2+1
                rollout_signals = td_vae_model.rollout(states, t1, t2, device)

                preds.append(rollout_signals.squeeze(0))
                next_states = next_states.squeeze(1)
                # Take the next_states for t1 ... t2+1
                gt.append(next_states[t1:t2+1])

                t1 += 1
                t2 += 1

        preds = torch.stack(preds)
        gt = torch.stack(gt).cuda()
        error = (preds - gt)**2

    preds = preds.cpu().numpy()
    path = 'td_vae_results_cartpole'
    pickle.dump(preds, open(path, 'wb'))


    print("Mean Error: {}".format(error.mean(0)[0].cpu().numpy()))
    print("Min  Error: {}".format(error.min(0)[0].cpu().numpy()))
    print("Max  Error: {}".format(error.max(0)[0].cpu().numpy()))
