# Copyright (c) 2021 Rui Shu
import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)

        # Sample from the encoder
        z_mean, z_logvar = self.enc(x)
        z = ut.sample_gaussian(z_mean, z_logvar)

        # Compute the log probabilities of the samples under the prior and the encoder
        log_pz = ut.log_normal_mixture(z, prior[0], prior[1]).sum(dim=-1)
        log_qz_x = ut.log_normal(z, z_mean, z_logvar).sum(dim=-1)

        # Sample from the decoder
        px_z = self.dec(z)

        # Compute the KL and reconstruction terms of the ELBO
        rec = -ut.log_bernoulli_with_logits(x, px_z).mean()
        kl = (log_qz_x - log_pz).mean()
        
        # Compute the negative ELBO bound
        nelbo = kl + rec

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
        # Outputs should all be scalar
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)

        # Sample from the encoder
        z_mean, z_logvar = self.enc(x)
        z_mean = ut.duplicate(z_mean.unsqueeze(0), iw)
        z_logvar = ut.duplicate(z_logvar.unsqueeze(0), iw)
        x = ut.duplicate(x.unsqueeze(0), iw)
        z = ut.sample_gaussian(z_mean, z_logvar)
        prior_mean = ut.duplicate(prior[0].unsqueeze(0), iw)
        prior_var = ut.duplicate(prior[1].unsqueeze(0), iw)

        # Compute the log probabilities of the samples under the prior and the encoder
        log_pz = ut.log_normal_mixture(z, prior_mean, prior_var).sum(dim=-1)
        log_qz_x = ut.log_normal(z, z_mean, z_logvar).sum(dim=-1)

        # Sample from the decoder
        logits = self.dec(z)

        rec = -ut.log_bernoulli_with_logits(x, logits)
        kl = log_qz_x - log_pz
        
        # Compute the negative ELBO bound
        nelbo = kl + rec
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
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
