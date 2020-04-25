import torch
import torch.nn as nn
import torch.nn.functional as F


#----------------------------------------------------------------------
# Smaller Modules
#----------------------------------------------------------------------


class MiniBatchStd(nn.Module):

    """
    Encourages generator to increase variation.
    """

    def __init__(self):
        super(MiniBatchStd, self).__init__()

    def forward(self, x, group_size=16):

        #group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        # s = x.shape                                             # [NCHW]  Input shape.
        # y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        # y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        # y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        # y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        # y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        # y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        # y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        # y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
        # return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

        group_size = min(group_size, x.shape[0])
        s = x.shape
        y = x.view(group_size, -1, s[1], s[2], s[3]).float()
        y -= torch.mean(y, axis=0, keepdim=True)
        y = torch.mean(y ** 2, axis=0)
        y = torch.sqrt(y + 1e-8)
        y = torch.mean(y, axis=[1, 2, 3], keepdim=True)
        y = y.repeat(group_size, 1, 1, 1)
        y = y.expand(-1, 1, s[2], s[3])
        return y  # or combine here to return new fmap

        # std = torch.std(x, dim=[0, 1])
        # return std.expand(x.shape[0], 1, -1, -1)  # returns shape N1HW (with C of 1)


class PixelNorm(nn.Module):

    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        # x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


#----------------------------------------------------------------------
# Medium modules
#----------------------------------------------------------------------


class StartingGeneratorStack(nn.Module):

    def __init__(self, resolution=4):
        super(StartingGeneratorStack, self).__init__()
        self.resolution = resolution

        # layers
        self.dense = nn.Linear(int(min(8192 / resolution, 512)),
                               int(min(8192 / resolution, 512)) * 16)

        # output 512 × 4 × 4
        self.conv1 = nn.Conv2d(in_channels=512,
                               out_channels=512,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=(1, 1))

        self.pixelnorm = PixelNorm()

        # remember out channels
        self.out_channels = self.conv1.out_channels

    def forward(self, x):

        # block 1
        x = self.dense(x)
        x = x.reshape(-1, 512, 4, 4)
        x = self.pixelnorm(F.leaky_relu(x, negative_slope=0.2))

        # block 2
        x = self.pixelnorm(F.leaky_relu(self.conv1(x), negative_slope=0.2))

        return x


class IntermediateGeneratorStack(nn.Module):

    def __init__(self, resolution=8, in_channels=512):
        super(IntermediateGeneratorStack, self).__init__()
        self.resolution = resolution
        self.in_channels = in_channels

        # layers
        self.upscale = nn.Upsample(scale_factor=2., mode='nearest')
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=int(min(8192 / resolution, 512)),
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=int(min(8192 / resolution, 512)),
                               out_channels=int(min(8192 / resolution, 512)),
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1)
        self.pixelnorm = PixelNorm()

        # remember out channels
        self.out_channels = self.conv2.out_channels

    def forward(self, x):
        x = self.upscale(x)

        # block 1
        x = self.pixelnorm(F.leaky_relu(self.conv1(x), negative_slope=0.2))

        # block 2
        x = self.pixelnorm(F.leaky_relu(self.conv2(x), negative_slope=0.2))

        return x


class FinalDiscriminatorStack(nn.Module):

    def __init__(self, resolution=4):
        super(FinalDiscriminatorStack, self).__init__()
        self.resolution = resolution

        # layers
        self.mini_batch = MiniBatchStd()

        self.conv1 = nn.Conv2d(in_channels=int(min(8192 / resolution, 512)) + 1,
                               out_channels=512,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=512,
                               out_channels=512,
                               kernel_size=(4, 4),
                               stride=1,
                               padding=1)

        # self.dense1 = nn.Linear(int(min(8192 / resolution, 512)) * 16,
        #                         min(int(8192 / resolution - 1), 512))

        self.dense2 = nn.Linear(512 * 3 * 3, 1)

        # remember out channels
        self.in_channels = self.conv1.in_channels - 1  # remove extra channel added by minibatchstd

    def forward(self, x):

        new_channel = self.mini_batch(x)
        x = torch.cat([x, new_channel], dim=1)

        # block 1
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = x.reshape(-1, 512 * 3 * 3)

        # block 2
        # x = F.leaky_relu(self.dense1(x), negative_slope=0.2)

        # block 3
        x = self.dense2(x)

        return x


class IntermediateDiscriminatorStack(nn.Module):

    def __init__(self, resolution=1024, out_channels=512):
        super(IntermediateDiscriminatorStack, self).__init__()
        self.resolution = resolution
        self.out_channels = out_channels

        # layers
        self.conv1 = nn.Conv2d(in_channels=int(min(8192 / (resolution * 2), 512)),
                               out_channels=int(min(8192 / resolution, 512)),
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=int(min(8192 / resolution, 512)),
                               out_channels=self.out_channels,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1)

        self.downscale = nn.AvgPool2d(kernel_size=(2, 2))

        self.in_channels = self.conv1.in_channels

    def forward(self, x):

        # block 1
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)

        # block 2
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)

        x = self.downscale(x)

        return x


#----------------------------------------------------------------------
# Larger modules
#----------------------------------------------------------------------


class Generator(nn.Module):

    def __init__(self, start_resolution=4):
        super(Generator, self).__init__()
        self.resolution = start_resolution
        start_stack = StartingGeneratorStack(resolution=4)
        self.equalize_learning(start_stack)
        self.modules_ = nn.ModuleList([start_stack])
        self.next_in_channels = self.modules_[-1].out_channels
        self.to_rgb = nn.Conv2d(in_channels=self.modules_[-1].out_channels,
                                out_channels=3,
                                kernel_size=(1, 1),
                                stride=1,
                                padding=0)
        # self.equalize_learning(self.to_rgb)
        self.prior_to_rgb = None

    def equalize_learning(self, new_module):
        # use equalized learning rate
        for module in new_module.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                shape = torch.tensor(module.weight.data.shape)
                if len(shape) == 4:
                    prod = torch.prod(shape[1:])         # leave out num out channels (feature maps)
                else:                                    # len(shape) == 2
                    prod = torch.prod(shape[1:])         # leave out num out channels (feature maps)
                he_init = torch.sqrt(torch.tensor(2.)) / torch.sqrt(prod.float())
                module.weight.data = torch.randn(module.weight.data.shape) * he_init
                module.bias.data = torch.zeros(module.bias.data.shape)

    def forward(self, x, alpha=1.):

        # go through the modules
        to_rgb = False
        for i, module in enumerate(self.modules_):
            if alpha < 1. and len(self.modules_) > 1 and (i == len(self.modules_) - 1):
                old_x = x * (1 - alpha)
                old_x = self.prior_to_rgb(old_x)
                old_x = module.upscale(old_x)
                new_x = module(x) * alpha
                new_x = self.to_rgb(new_x)
                x = new_x + old_x
                to_rgb = True
            else:
                x = module(x)
        if not to_rgb:
            x = self.to_rgb(x)
        return torch.clamp(x, min=0., max=1.)

    def increase_resolution(self):
        self.resolution *= 2

        new_module = IntermediateGeneratorStack(resolution=int(self.resolution),
                                                in_channels=self.next_in_channels)
        self.equalize_learning(new_module)
        self.next_in_channels = new_module.out_channels
        self.modules_.append(new_module)

        self.prior_to_rgb = self.to_rgb

        self.to_rgb = nn.Conv2d(in_channels=self.modules_[-1].out_channels,
                                out_channels=3,
                                kernel_size=(1, 1),
                                stride=1,
                                padding=0)

        if self.to_rgb.in_channels == self.prior_to_rgb.in_channels:
            self.to_rgb.weight.data = self.prior_to_rgb.weight.data
        else:
            approximate_decrease = self.prior_to_rgb.in_channels // self.to_rgb.in_channels
            new_weight = torch.zeros(self.to_rgb.weight.data.shape)
            for n in range(self.to_rgb.weight.data.shape[0]):
                for new_c, c in enumerate(range(0, self.to_rgb.weight.data.shape[1], approximate_decrease)):
                    mean = torch.mean(self.prior_to_rgb.weight.data[n, c:c + approximate_decrease, :, :],
                                      dim=0)
                    new_weight[n, new_c] = mean
            self.to_rgb.weight.data = new_weight

    @property
    def newest_params(self):
        return self.modules_[-1].parameters()


class Discriminator(nn.Module):

    def __init__(self, start_resolution=4):
        super(Discriminator, self).__init__()
        self.resolution = start_resolution
        final_stack = FinalDiscriminatorStack(resolution=4)
        self.equalize_learning(final_stack)
        self.modules_ = nn.ModuleList([final_stack])
        self.from_rgb = nn.Conv2d(in_channels=3,
                                  out_channels=final_stack.in_channels,
                                  kernel_size=(1, 1),
                                  stride=1,
                                  padding=0)
        # self.equalize_learning(self.from_rgb)
        self.downscale = nn.AvgPool2d(kernel_size=(2, 2))
        self.prior_from_rgb = None

    def equalize_learning(self, new_module):
        # use equalized learning rate
        for module in new_module.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                shape = torch.tensor(module.weight.data.shape)
                if len(shape) == 4:
                    prod = torch.prod(shape[1:])         # leave out num out channels (feature maps)
                else:                                    # len(shape) == 2
                    prod = torch.prod(shape[1:])         # leave out num out channels (feature maps)
                he_init = torch.sqrt(torch.tensor(2.)) / torch.sqrt(prod.float())
                module.weight.data = torch.randn(module.weight.data.shape) * he_init
                module.bias.data = torch.zeros(module.bias.data.shape)

    def forward(self, x, alpha=1.):

        # go through the modules
        for i, module in enumerate(self.modules_):
            if alpha < 1. and len(self.modules_) > 1 and i == 0:
                old_x = self.downscale(x)
                old_x = F.leaky_relu(self.prior_from_rgb(old_x), negative_slope=0.2)
                old_x = old_x * (1 - alpha)
                new_x = F.leaky_relu(self.from_rgb(x), negative_slope=0.2)
                new_x = module(new_x) * alpha
                x = new_x + old_x
            elif i == 0:
                x = self.from_rgb(x)
                x = module(x)
            else:
                x = module(x)

        return x

    def increase_resolution(self):
        self.resolution *= 2

        new_module = IntermediateDiscriminatorStack(resolution=int(self.resolution),
                                                    out_channels=int(min(8192 / self.resolution, 512)))
        self.equalize_learning(new_module)
        self.modules_.insert(0, new_module)

        self.prior_from_rgb = self.from_rgb

        self.from_rgb = nn.Conv2d(in_channels=3,
                                  out_channels=self.modules_[0].in_channels,
                                  kernel_size=(1, 1),
                                  stride=1,
                                  padding=0)

    @property
    def newest_params(self):
        return self.modules_[0].parameters()
