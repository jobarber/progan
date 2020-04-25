import torch


def generator_criterion(discriminator_fake_output):
    # def G_wgan_acgan(G, D, opt, training_set, minibatch_size,
    #     cond_weight = 1.0): # Weight of the conditioning term.
    #
    #     latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    #     labels = training_set.get_random_labels_tf(minibatch_size)
    #     fake_images_out = G.get_output_for(latents, labels, is_training=True)
    #     fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    #     loss = -fake_scores_out
    #
    #     if D.output_shapes[1][1] > 0:
    #         with tf.name_scope('LabelPenalty'):
    #             label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
    #         loss += label_penalty_fakes * cond_weight
    #     return loss
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

    #     # Apply dynamic loss scaling for the given expression.
    #     def apply_loss_scaling(self, value):
    #         assert is_tf_expression(value)
    #         if not self.use_loss_scaling:
    #             return value
    #         return value * exp2(self.get_loss_scaling_var(value.device))
    #
    #     # Undo the effect of dynamic loss scaling for the given expression.
    #     def undo_loss_scaling(self, value):
    #         assert is_tf_expression(value)
    #         if not self.use_loss_scaling:
    #             return value
    #         return value * exp2(-self.get_loss_scaling_var(value.device))

    mixed_loss = torch.sum(mixed_scores_out)  # originally wrapped in loss_scaling, but appears to not use it
    grad_outputs = torch.ones(mixed_loss.size()).to(device)
    mixed_grads = torch.autograd.grad(mixed_loss, mixed_images_out, grad_outputs=grad_outputs,
                                      create_graph=True)[0]
    mixed_norms = torch.sqrt(torch.sum(mixed_grads ** 2, axis=[1, 2, 3]))
    gradient_penalty = ((mixed_norms - wgan_target) ** 2).reshape(-1, 1)

    loss += gradient_penalty * (wgan_lambda / (wgan_target ** 2))
    epsilon_penalty = real_scores_out ** 2
    loss += epsilon_penalty * wgan_epsilon

    # def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels,
    #     wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    #     wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    #     wgan_target     = 1.0,      # Target value for gradient magnitudes.
    #     cond_weight     = 1.0):     # Weight of the conditioning terms.
    #
    #     latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    #     fake_images_out = G.get_output_for(latents, labels, is_training=True)
    #     real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    #     fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    #     real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    #     fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    #     loss = fake_scores_out - real_scores_out
    #
    #     with tf.name_scope('GradientPenalty'):
    #         mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
    #         mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
    #         mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
    #         mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
    #         mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
    #         mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
    #         mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
    #         mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
    #         gradient_penalty = tf.square(mixed_norms - wgan_target)
    #     loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    #
    #     with tf.name_scope('EpsilonPenalty'):
    #         epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    #     loss += epsilon_penalty * wgan_epsilon
    #
    #     if D.output_shapes[1][1] > 0:
    #         with tf.name_scope('LabelPenalty'):
    #             label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
    #             label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
    #             label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
    #             label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
    #         loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    #     return loss

    return loss.mean()  # added mean in pytorch implementation; tf did not have!

    # gradient_penalty = get_gradient_penalty(discriminator, generated_output, real_output, alpha=alpha, lambda_=lambda_)
    # discriminator_real_output = discriminator(real_output, alpha=alpha)
    # discriminator_fake_output = discriminator(generated_output, alpha=alpha)

    # return gradient penalty added to normal loss
    # return -torch.mean(discriminator_real_output) + torch.mean(discriminator_fake_output) + gradient_penalty


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
    # gradients = gradients.view(batch_size, -1)

    # with tf.name_scope('GradientPenalty'):
    #     #         mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
    #     #         mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
    #     #         mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
    #     #         mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
    #     #         mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
    #     #         mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
    #     #         mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
    #     #         mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
    #     #         gradient_penalty = tf.square(mixed_norms - wgan_target)
    #     #     loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    # get gradient penalty
    mixed_norms = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2, 3]))
    gradient_penalty = (mixed_norms - 1) ** 2
    # gradient_penalty = gradient_penalty.item()

    # remove gradient tracking
    del interpolation
    torch.cuda.empty_cache()

    return (gradient_penalty * (lambda_ / 1. ** 2)).mean()
