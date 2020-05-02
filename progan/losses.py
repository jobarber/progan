import torch


def generator_criterion(discriminator_fake_output):
    return -torch.mean(discriminator_fake_output)


def discriminator_criterion(G, D, reals, alpha=1., device='cuda',
                            wgan_lambda=10.0,  # Weight for the gradient penalty term.
                            wgan_epsilon=0.001,  # Weight for the epsilon term, \epsilon_{drift}.
                            wgan_target=1.0):  # Target value for gradient magnitudes.
    """
    Wasserstein distance criterion.

    Parameters
    ----------
    discriminator_fake_output
    discriminator_real_output
    generated_output
    real_output
    discriminator
    lambda_

    Returns
    -------

    """
    latents = torch.randn([reals.shape[0], 512]).to(device)
    fake_images_out = G(latents)
    real_scores_out = D(reals, alpha=alpha)
    fake_scores_out = D(fake_images_out, alpha=alpha)
    loss = fake_scores_out - real_scores_out

    mixing_factors = torch.rand([reals.shape[0], 1, 1, 1]).to(device)
    mixed_images_out = torch.lerp(reals, fake_images_out, mixing_factors).to(device)
    mixed_scores_out = D(mixed_images_out, alpha=alpha)

    mixed_loss = torch.sum(mixed_scores_out)  # originally wrapped in loss_scaling, but appears to not use it
    grad_outputs = torch.ones(mixed_loss.size()).to(device)
    mixed_grads = torch.autograd.grad(mixed_loss, mixed_images_out, grad_outputs=grad_outputs,
                                      create_graph=True)[0]
    mixed_norms = torch.sqrt(torch.sum(mixed_grads ** 2, axis=[1, 2, 3]))
    gradient_penalty = ((mixed_norms - wgan_target) ** 2).reshape(-1, 1)

    loss += gradient_penalty * (wgan_lambda / (wgan_target ** 2))
    epsilon_penalty = real_scores_out ** 2
    loss += epsilon_penalty * wgan_epsilon

    return loss.mean()


def get_gradient_penalty(discriminator, generated_output, real_output, alpha=1., lambda_=10.):

    """
    Get gradient penalty.

    Parameters
    ----------
    discriminator
    generated_output
    real_output
    lambda_

    Returns
    -------

    """

    if real_output.shape != generated_output.shape:
        generated_output = generated_output[:real_output.shape[0]]

    batch_size = real_output.shape[0]

    # get epsilon
    # each image receives its own epsilon
    # (e.g., image 1 eps == .8047, image 2 eps == .1988, etc.)
    epsilon = torch.rand(batch_size, 1, 1, 1)
    # stretch the eps value to dim of each image
    epsilon = epsilon.expand(real_output.shape)
    epsilon = epsilon.cuda()

    # get interpolation
    interpolation = epsilon * real_output.data + (1 - epsilon) * generated_output.data
    interpolation.requires_grad = True
    interpolation = interpolation.cuda()

    # get interpolation logits
    interpolation_logits = discriminator(interpolation, alpha=alpha)

    # get gradients
    grad_outputs = torch.ones(interpolation_logits.size())
    grad_outputs = grad_outputs.cuda()
    gradients = torch.autograd.grad(outputs=interpolation_logits,
                                    inputs=interpolation,
                                    grad_outputs=grad_outputs,
                                    create_graph=True,
                                    retain_graph=True)[0]
    gradients = gradients.detach()

    # get gradient penalty
    mixed_norms = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2, 3]))
    gradient_penalty = (mixed_norms - 1) ** 2
    # gradient_penalty = gradient_penalty.item()

    # remove gradient tracking
    del interpolation
    torch.cuda.empty_cache()

    return (gradient_penalty * (lambda_ / 1. ** 2)).mean()
