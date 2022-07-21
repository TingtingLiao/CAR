import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxMixturePosePrior(nn.Module):
    def __init__(self, n_gaussians=8, prefix=3, device=torch.device('cpu')):
        super(MaxMixturePosePrior, self).__init__()
        self.prefix = prefix
        self.n_gaussians = n_gaussians
        self.create_prior_from_cmu(device)

    def create_prior_from_cmu(self, device):
        """Load the gmm from the CMU motion database."""
        with open(os.path.join(os.path.dirname(__file__), '../../data/smpl_related/smpl_data/gmm_08.pkl'), 'rb') as f:
            gmm = pickle.load(f, encoding='bytes')
        precs = np.asarray([np.linalg.cholesky(np.linalg.inv(cov)) for cov in gmm[b'covars']])
        means = np.asarray(gmm[b'means'])  # [8, 69]

        sqrdets = np.array([(np.sqrt(np.linalg.det(c))) for c in gmm[b'covars']])
        const = (2 * np.pi) ** (69 / 2.)
        weights = np.asarray(gmm[b'weights'] / (const * (sqrdets / sqrdets.min())))

        self.precs = torch.from_numpy(precs).to(device)  # [8, 69, 69]
        self.means = torch.from_numpy(means).to(device)  # [8, 69]
        self.weights = torch.from_numpy(weights).to(device)

    def forward(self, pose):
        # assert pose.dim() == 2 amd
        theta = pose[:, self.prefix:]
        batch, dim = theta.shape
        theta = theta.expand(self.n_gaussians, batch, dim).permute(1, 0, 2)
        theta = (theta - self.means[None])[:, :, None, :]
        loglikelihoods = np.sqrt(0.5) * torch.matmul(theta, self.precs.expand(batch, *self.precs.shape)).squeeze(2)
        results = (loglikelihoods * loglikelihoods).sum(-1) - self.weights.log()
        return results.min()


def PoseAngleConstrain(theta):
    loss = torch.exp(theta[:, 55]) + torch.exp(-theta[:, 58]) + torch.exp(-theta[:, 12]) + torch.exp(-theta[:, 15]) + theta[:, [56, 59]].abs()

    return torch.mean(loss ** 2)


