from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import mixture
from torchvision.datasets import ImageFolder
from torchvision import transforms


class ImageFolderDataset(ImageFolder):

    def __init__(self, root='CelebA-HQ-img', resolution=4, length=None, sample_limit=None, **kwargs):

        """
        Constructor.

        root (string) – Root directory path.

        transform (callable, optional) – A function/transform that takes in an PIL image
                                         and returns a transformed version. E.g, transforms.RandomCrop

        target_transform (callable, optional) – A function/transform that takes in the target and
                                                transforms it.

        loader (callable, optional) – A function to load an image given its path.

        is_valid_file – A function that takes path of an Image file and check if the file is
                        a valid file (used to check of corrupt files)

        pixels (int) – Number of pixels of width and height (assumes a square image).
        """
        self.length = length
        transform = transforms.Compose([
            transforms.Resize(size=(resolution, resolution)),
            transforms.ColorJitter(brightness=0.005, contrast=0., saturation=0., hue=0.),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomPerspective(distortion_scale=0.005,
                                         p=0.05),
            transforms.RandomRotation(0.2),
            transforms.ToTensor()
        ])
        feeder_kwargs = {'root': root, 'transform': transform}
        feeder_kwargs.update(kwargs)

        super(ImageFolderDataset, self).__init__(feeder_kwargs.pop('root'), **feeder_kwargs)
        if sample_limit:
            self.samples = self.samples[:sample_limit]

    def __getitem__(self, item):
        return super(ImageFolderDataset, self).__getitem__(item % len(self.samples))

    def __len__(self):
        if self.length:
            return self.length
        return super(ImageFolderDataset, self).__len__()


class ImageFolderClusteredDataset(ImageFolder):

    def __init__(self, root='CelebA-HQ-img', resolution=4, length=None, sample_limit=None,
                 category=0, **kwargs):

        """
        Constructor.

        root (string) – Root directory path.

        transform (callable, optional) – A function/transform that takes in an PIL image
                                         and returns a transformed version. E.g, transforms.RandomCrop

        target_transform (callable, optional) – A function/transform that takes in the target and
                                                transforms it.

        loader (callable, optional) – A function to load an image given its path.

        is_valid_file – A function that takes path of an Image file and check if the file is
                        a valid file (used to check of corrupt files)

        pixels (int) – Number of pixels of width and height (assumes a square image).
        """
        self.length = length
        transform = transforms.Compose([
            transforms.Resize(size=(resolution, resolution)),
            transforms.ColorJitter(brightness=0.005, contrast=0., saturation=0., hue=0.),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomPerspective(distortion_scale=0.005,
                                         p=0.05),
            transforms.RandomRotation(0.2),
            transforms.ToTensor()
        ])
        feeder_kwargs = {'root': root, 'transform': transform}
        feeder_kwargs.update(kwargs)

        super(ImageFolderDataset, self).__init__(feeder_kwargs.pop('root'), **feeder_kwargs)
        if sample_limit:
            self.samples = self.samples[:sample_limit]

        self.category = category

        self.category_image_indices = defaultdict(list)

        self.latent_clusterer = mixture.GaussianMixture(n_components=21)
        self._fit_latent_clusterer()

        self.image_clusterer = mixture.GaussianMixture(n_components=21)
        self._fit_image_clusterer()

    def __getitem__(self, item):
        viable_samples = self.category_image_indices[self.category]
        return super(ImageFolderDataset, self).__getitem__(viable_samples[item] % len(viable_samples))

    def __len__(self):
        if self.length:
            return self.length
        return super(ImageFolderDataset, self).__len__()

    def _fit_image_clusterer(self):
        orb = cv2.ORB_create(nfeatures=50, scaleFactor=1.5)
        granular_samples = dict()

        for i in range(len(self.samples)):

            sample, _ = super(ImageFolderDataset, self).__getitem__(i % len(self.samples))

            sample = F.interpolate(sample.unsqueeze(0), size=128).squeeze(0)

            image = sample.permute(1, 2, 0).numpy() * 256
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp = orb.detect(image, None)
            kp, des = orb.compute(image, kp)
            kp = sorted(kp, key=lambda x: x.response, reverse=False)[:3]
            granular_samples[i] = torch.zeros((3, 16, 16))

            for j, k in enumerate(kp):
                height, width = k.pt
                height, width = int(height), int(width)

                if width < 8:
                    startw = 0
                    endw = startw + 16
                elif width + 8 >= gray.shape[1]:
                    startw = gray.shape[1] - 16
                    endw = startw + 16
                else:
                    startw = width - 8
                    endw = width + 8

                if height < 8:
                    starth = 0
                    endh = starth + 16
                elif height + 8 >= gray.shape[1]:
                    starth = gray.shape[1] - 16
                    endh = starth + 16
                else:
                    starth = height - 8
                    endh = height + 8

                snippet = torch.tensor(gray[starth:endh, startw:endw])
                granular_samples[i][j] = snippet

        granular_tensor = torch.zeros((len(granular_samples), 3 * 16 * 16))
        for key in granular_samples:
            granular_tensor[key] = granular_samples[key].reshape(-1)

        normed_samples = granular_tensor.numpy() / 256.

        self.image_clusterer.fit(normed_samples)

        for i in range(0, granular_tensor.shape[0], 20):
            categories = self.image_clusterer.predict(granular_tensor[i:i + 20])
            for j, category in enumerate(categories):
                self.category_image_indices[category].append(i + j)

    def _fit_latent_clusterer(self):
        self.latent_clusterer.fit(np.random.randn(10_000, 512))

    def set_category(self, num):
        self.category = num
