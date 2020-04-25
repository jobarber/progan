import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from progan.datasets import ImageFolderDataset
from progan.losses import discriminator_criterion, generator_criterion
from progan.models import Discriminator, Generator


def train(transition_every=1_300_000, dmodelpath=None, gmodelpath=None):  # paper transitioned every 800k

    if gmodelpath and dmodelpath:
        generator = torch.load('modeldata/' + gmodelpath).cuda()
        discriminator = torch.load('modeldata/' + dmodelpath).cuda()

        resolution = generator.resolution
        alpha = torch.tensor(0.).cuda()
    else:
        generator = Generator(start_resolution=4).cuda()
        discriminator = Discriminator(start_resolution=4).cuda()

        resolution = 4
        alpha = torch.tensor(1.).cuda()

    print(generator, '\n')

    print(discriminator, '\n')

    max_alpha = torch.tensor(1.).cuda()

    dataset = ImageFolderDataset(root='downloads', resolution=generator.resolution, length=16_000,
                                 sample_limit=None)
    dataloader = DataLoader(dataset, batch_size=min(8192 // (resolution * 2), 512))

    #  α = 0.001, β1 = 0, β2 = 0.99, and eps = 10−8
    generator_optimizer = optim.Adam(params=generator.parameters(), lr=1e-4, betas=(0., 0.9), eps=1e-8)
    # generator_optimizer = optim.RMSprop(params=generator.parameters(), lr=5e-5)
    discriminator_optimizer = optim.Adam(params=discriminator.parameters(), lr=1e-4, betas=(0., 0.9), eps=1e-8, )
    # discriminator_optimizer = optim.RMSprop(params=discriminator.parameters(), lr=5e-5)

    steady_sample = torch.randn(16, 512).cuda()

    counter = 0
    seen_images = 0

    for epoch in range(30_000):

        d_running_loss = 0.
        g_running_loss = 0.

        for batch_index, batch in enumerate(dataloader):

            alpha = min(alpha + dataloader.batch_size / transition_every, max_alpha)

            X, y = batch
            if X.shape[0] < dataloader.batch_size:
                continue
            X = X.cuda()

            if batch_index == 0:
                sample = generator(steady_sample, alpha=alpha)
                res_nums = {1024: 2, 512: 2, 256: 4, 128: 6, 64: 8, 32: 12, 16: 12, 8: 16, 4: 16}
                res_rows = {1024: 2, 512: 2, 256: 4, 128: 6, 64: 2, 32: 3, 16: 4, 8: 4, 4: 4}
                sample = torch.cat([sample[:res_nums.get(resolution, 2) // 2], X[:16]], dim=0)[
                         :res_nums.get(resolution, 2)]
                # image = make_grid(sample, nrow=2 if sample.shape[0] >= 4 else 1)
                save_image(sample, 'inferences/inference{}_{}.png'.format(epoch, seen_images),
                           nrow=res_rows.get(resolution, 2))
                print('fake sample =>', sample[0])
                # print('real sample =>', sample[-1])

            ###################

            # train discriminator (5x as much for generic Wasserstein)

            for param in discriminator.parameters():
                param.requires_grad = True

            for _ in range(2):
                discriminator_optimizer.zero_grad()

                discriminator_loss = discriminator_criterion(generator, discriminator, X, alpha=alpha)

                d_running_loss += discriminator_loss

                discriminator_loss.backward()

                # # not needed: gradient penalty replaces clamping
                # # clamp for generic Wasserstein distance
                # # see: https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
                # for p in discriminator.parameters():
                #     p.data.clamp_(-0.01, 0.01)

                discriminator_optimizer.step()

            # train generator

            generator_optimizer.zero_grad()

            X_generator = torch.randn((dataloader.batch_size, 512)).to('cuda')

            for param in discriminator.parameters():
                param.requires_grad = False

            generator_output = generator(X_generator, alpha=alpha)
            outputs = discriminator(generator_output, alpha=alpha)
            generator_loss = generator_criterion(outputs)

            g_running_loss += generator_loss

            generator_loss.backward()
            generator_optimizer.step()

            seen_images += dataloader.batch_size

            #################

            print('Epoch: {}, Seen Images: {}, Alpha: {} -- GLoss: {}, DLoss: {}, GRunLoss: {}, DRunLoss: {}'
                  .format(epoch, seen_images, alpha, generator_loss, discriminator_loss,
                          g_running_loss / (batch_index + 1), d_running_loss / (batch_index + 1)))

            if 0 <= seen_images % 1_000 < dataloader.batch_size:
                sample = generator(steady_sample, alpha=alpha)
                res_nums = {1024: 2, 512: 2, 256: 4, 128: 6, 64: 8, 32: 12, 16: 12, 8: 16, 4: 16}
                res_rows = {1024: 2, 512: 2, 256: 4, 128: 6, 64: 2, 32: 3, 16: 4, 8: 4, 4: 4}
                sample = torch.cat([sample[:res_nums.get(resolution, 2) // 2], X[:16]], dim=0)[
                         :res_nums.get(resolution, 2)]
                save_image(sample, 'inferences/inference{}_{}.png'.format(epoch, seen_images),
                           nrow=res_rows.get(resolution, 2))
                # print('real sample =>', sample[-1])
                print('fake sample =>', sample[0])

            # transition every other
            if (seen_images > transition_every // 2 and
                    0 <= seen_images % (transition_every * 2) - transition_every < dataloader.batch_size):

                torch.save(generator, 'modeldata/gmodel{}_{}.pt'.format(resolution, g_running_loss))
                torch.save(discriminator, 'modeldata/dmodel{}_{}.pt'.format(resolution, g_running_loss))

                resolution *= 2
                print('increasing resolution to', resolution)
                dataset = ImageFolderDataset(root='downloads', resolution=resolution, length=16_000,
                                             sample_limit=None)
                dataloader = DataLoader(dataset, batch_size=min(8192 // (resolution * 2), 512))

                generator.increase_resolution()
                discriminator.increase_resolution()

                generator = generator.cuda()
                discriminator = discriminator.cuda()

                generator_optimizer.add_param_group({'params': generator.newest_params, 'lr': 5e-5})
                discriminator_optimizer.add_param_group({'params': discriminator.newest_params, 'lr': 5e-5})

                print(generator)
                print(discriminator)

                alpha = torch.tensor(0.).cuda()
                break  # cannot continue with current set of lower res batches from dataloader


if __name__ == '__main__':
    # train(transition_every=800_000,
    #       dmodelpath='dmodel8_2.632054090499878.pt',
    #       gmodelpath='gmodel8_2.632054090499878.pt')
    g = train(transition_every=800_000,
              dmodelpath=None,
              gmodelpath=None)
