import torch
from torchvision.utils import save_image


def make_inference(gmodelpath, num_inferences=32):
    generator = torch.load('modeldata/' + gmodelpath).cuda()
    with torch.no_grad():
        for i in range(num_inferences):
            latent = torch.randn((1, 512)).cuda()
            sample = generator(latent, alpha=1.)
            save_image(sample, 'inferences/real_inference{}.png'.format(i))


if __name__ == '__main__':
    make_inference('gmodel32_744.3274536132812.pt', num_inferences=256)
