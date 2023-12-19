# Copyright (c) 2021 Rui Shu

import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): average Negative evidence lower bound
            kl: tensor: (): average ELBO KL divergence to prior
            rec: tensor: (): average ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        batch_size = x.size(0)
        z_mean, z_logvar = self.enc(x)
        z = ut.sample_gaussian(z_mean, z_logvar)
        logits = self.dec(z)
        rec = F.binary_cross_entropy_with_logits(logits, x, reduction='sum') / batch_size
        kl = ut.kl_normal(z_mean, z_logvar, self.z_prior[0], self.z_prior[1]).sum() / batch_size
        nelbo = rec + kl
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Note that niwae = -log(1/I * sum_{i=1}^I w_i), where w_i = p(x, z_i) / q(z_i | x)
        #
        # Outputs should all be scalar
        ################################################################################
        z_mean, z_logvar = self.enc(x)
        z_mean = ut.duplicate(z_mean.unsqueeze(0), iw)
        z_logvar = ut.duplicate(z_logvar.unsqueeze(0), iw)
        x = ut.duplicate(x.unsqueeze(0), iw)
        z = ut.sample_gaussian(z_mean, z_logvar)
        logits = self.dec(z)
        rec = -ut.log_bernoulli_with_logits(x, logits)
        kl = ut.kl_normal(z_mean, z_logvar, self.z_prior[0], self.z_prior[1])
        nelbo = rec + kl
        niwae = -ut.log_mean_exp(-nelbo, dim=0).mean(axis=0)
        kl = kl.mean()
        rec = rec.mean()

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
