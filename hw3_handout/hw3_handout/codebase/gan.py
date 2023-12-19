import torch
from torch.nn import functional as F


def loss_nonsaturating_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR DISCRIMINATOR CODE STARTS HERE
    # Generate fake images
    x_fake = g(z)
    
    # Compute discriminator outputs for real and fake images
    logits_real = d(x_real)
    logits_fake = d(x_fake)
    
    # Compute losses for real and fake images
    loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
    
    # Combine losses
    d_loss = loss_real + loss_fake
    
    # YOUR CODE ENDS HERE

    return d_loss

def loss_nonsaturating_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR GENERATOR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - F.logsigmoid
    x_fake = g(z)

    # Compute discriminator output on fake images
    logits_fake = d(x_fake)

    # Compute the generator loss as the negative log-sigmoid of the discriminator's output
    g_loss = -F.logsigmoid(logits_fake).mean()

    # YOUR CODE ENDS HERE

    return g_loss


def conditional_loss_nonsaturating_d(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_fake = y_real  # use the real labels as the fake labels as well

    # YOUR DISCRIMINATOR STARTS HERE
    x_fake = g(z, y_fake)

    # Compute discriminator outputs for real and fake images
    logits_real = d(x_real, y_real)
    logits_fake = d(x_fake.detach(), y_fake)

    # Compute losses for real and fake images
    loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))

    # Combine losses
    d_loss = loss_real + loss_fake
    # YOUR CODE ENDS HERE

    return d_loss


def conditional_loss_nonsaturating_g(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_fake = y_real  # use the real labels as the fake labels as well

    # YOUR DISCRIMINATOR STARTS HERE
    x_fake = g(z, y_fake)

    # Compute discriminator output on fake images
    logits_fake = d(x_fake, y_fake)

    # Compute the generator loss as the negative log-sigmoid of the discriminator's output
    g_loss = -F.logsigmoid(logits_fake).mean()
    # YOUR CODE ENDS HERE

    return g_loss


def loss_wasserstein_gp_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - torch.rand
    #   - torch.autograd.grad(..., create_graph=True)
    x_fake = g(z)#.detach()

    # Discriminator outputs for real and fake data
    d_real = d(x_real)
    d_fake = d(x_fake)

    # d_loss_real = torch.mean(d_real)
    # d_loss_fake = torch.mean(d_fake)

    # Gradient penalty
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    x_hat = alpha * x_fake + (1 - alpha) * x_real
    d_hat = d(x_hat)

    gradients = torch.autograd.grad(outputs=d_hat.sum(), inputs=x_hat, #grad_outputs=torch.ones(d_hat.size(), device=device),
                     create_graph=True#, retain_graph=True, only_inputs=True
                     )[0]
    lambda_gp = 10
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp

    # Combine losses
    d_loss = d_fake - d_real + gradient_penalty
    d_loss = d_loss.mean()
    # YOUR CODE ENDS HERE

    return d_loss


def loss_wasserstein_gp_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): wasserstein generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    x_fake = g(z)

    # Discriminator output on fake data
    d_fake = d(x_fake)

    # WGAN generator loss
    g_loss = -torch.mean(d_fake)
    # YOUR CODE ENDS HERE

    return g_loss
