from torchvision.datasets import ImageFolder
from torchvision import transforms


class ImageFolderDataset(ImageFolder):

    def __init__(self, root='modified-downloads', resolution=4, length=None, sample_limit=None, **kwargs):

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
            # transforms.ColorJitter(brightness=0., contrast=0., saturation=0., hue=.1 / 256),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomPerspective(distortion_scale=0.005,
                                         p=0.05),
            transforms.RandomRotation(1),
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
