from __future__ import annotations
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import rand
from torch.optim import Adam
from torch.hub import load_state_dict_from_url
from deepinv.physics import Physics
from deepinv.loss import MCLoss
from .base import Reconstructor
from deepinv.utils.decorators import _deprecated_alias


class PatchGANDiscriminator(nn.Module):
    r"""PatchGAN Discriminator model.

    This discriminator model was originally proposed by :footcite:t:`isola2017image` and classifies whether each patch of an image is real
    or fake.

    Implementation adapted from :footcite:t:`kupyn2018deblurgan`.

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for how to use this for adversarial training.

    :param int input_nc: number of input channels, defaults to 3
    :param int ndf: hidden layer size, defaults to 64
    :param int n_layers: number of hidden conv layers, defaults to 3
    :param bool use_sigmoid: use sigmoid activation at end, defaults to False
    :param bool batch_norm: whether to use batch norm layers, defaults to True
    :param bool bias: whether to use bias in conv layers, defaults to True
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_sigmoid: bool = False,
        batch_norm: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        kw = 4  # kernel width
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=bias,
                ),
                nn.BatchNorm2d(ndf * nf_mult) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=bias,
            ),
            nn.BatchNorm2d(ndf * nf_mult) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: Tensor):
        r"""
        Forward pass of discriminator model.

        :param torch.Tensor x: input image
        """
        return self.model(x)


class ESRGANDiscriminator(nn.Module):
    r"""ESRGAN Discriminator.

    The ESRGAN discriminator model was originally proposed by :footcite:t:`wang2018esrgan`. Implementation taken from
    https://github.com/edongdongchen/EI/blob/main/models/discriminator.py.

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for how to use this for adversarial training.

    :param tuple img_size: shape of input image


    """

    @_deprecated_alias(input_shape="img_size")
    def __init__(self, img_size: tuple):
        super().__init__()
        self.img_size = img_size
        in_channels, in_height, in_width = self.img_size
        patch_h, patch_w = int(in_height / 2**4), int(in_width / 2**4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)
            )
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(
                nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(
                discriminator_block(in_filters, out_filters, first_block=(i == 0))
            )
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        r"""
        Forward pass of discriminator model.

        :param torch.Tensor x: input image
        """
        return self.model(x)


class RealESRGANDiscriminator(nn.Module):
    r"""Real-ESRGAN Discriminator.

    The Real-ESRGAN discriminator model was originally proposed by :footcite:t:`wang2021real`.
    Implementation taken from https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/discriminator_arch.py
    """

    def __init__(
        self,
        in_nc: int,
        num_feat: int | tuple = 64,
        skip_connection: bool = True,
        dim: int | str = 2,
        pretrained: str | None = None,
    ):
        super(RealESRGANDiscriminator, self).__init__()
        if isinstance(num_feat, int):
            num_feat = (num_feat, num_feat * 2, num_feat * 4, num_feat * 8)
        if len(num_feat) != 4:
            print(num_feat)
            print(len(num_feat))
            raise ValueError("num_feat must be an int or a tuple of length 4.")

        self.skip_connection = skip_connection
        norm = nn.utils.spectral_norm
        conv = {"2": nn.Conv2d, "3": nn.Conv3d}[str(dim)]
        self.conv0 = conv(in_nc, num_feat[0], kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(conv(num_feat[0], num_feat[1], 4, 2, 1, bias=False))
        self.conv2 = norm(conv(num_feat[1], num_feat[2], 4, 2, 1, bias=False))
        self.conv3 = norm(conv(num_feat[2], num_feat[3], 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(conv(num_feat[3], num_feat[2], 3, 1, 1, bias=False))
        self.conv5 = norm(conv(num_feat[2], num_feat[1], 3, 1, 1, bias=False))
        self.conv6 = norm(conv(num_feat[1], num_feat[0], 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(conv(num_feat[0], num_feat[0], 3, 1, 1, bias=False))
        self.conv8 = norm(conv(num_feat[0], num_feat[0], 3, 1, 1, bias=False))
        self.conv9 = conv(num_feat[0], 1, 3, 1, 1)

        if pretrained is not None:
            if pretrained not in ["x2", "x4"]:
                raise ValueError("pretrained must be one of [None, 'x2', 'x4']")
            if (
                dim != 2
                or num_feat != (64, 128, 256, 512)
                or in_nc != 3
                or not skip_connection
            ):
                raise ValueError(
                    "Pretrained models only available in 2D, with settings num_feat=(64,128,256,512), in_nc=3 and skip_connection=True"
                )
            url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_{pretrained}plus_netD.pth"
            state_dict = load_state_dict_from_url(
                url, progress=True, map_location="cpu", weights_only=True
            )
            self.load_state_dict(state_dict["params"], strict=True)

    def forward(self, x: Tensor) -> Tensor:
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


class DCGANDiscriminator(nn.Module):
    r"""DCGAN Discriminator.

    The DCGAN discriminator model was originally proposed by :footcite:t:`radford2015unsupervised`. Implementation taken from
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html.

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for how to use this for adversarial training.

    :param int ndf: hidden layer size, defaults to 64
    :param int nc: number of input channels, defaults to 3


    """

    def __init__(self, ndf: int = 64, nc: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor):
        r"""Forward pass of discriminator model.

        :param torch.Tensor x: input image
        """
        return self.model(x)


class DCGANGenerator(nn.Module):
    r"""DCGAN Generator.

    The DCGAN generator model was originally proposed by :footcite:t:`radford2015unsupervised`
    and takes a latent sample as input.

    Implementation taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for how to use this for adversarial training.

    :param int output_size: desired square size of output image. Choose from 64 or 128, defaults to 64
    :param int nz: latent dimension, defaults to 100
    :param int ngf: hidden layer size, defaults to 64
    :param int nc: number of image output channels, defaults to 3
    """

    def __init__(
        self, output_size: int = 64, nz: int = 100, ngf: int = 64, nc: int = 3
    ):
        super().__init__()
        self.nz = nz
        # input is (b, nz, 1, 1), output is (b, nc, output_size, output_size)
        if output_size == 64:
            layers = [
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
            ]
        elif output_size == 128:
            layers = [
                nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
            ]
        else:
            raise ValueError("output_size must be 64 or 128.")

        layers += [
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, z: Tensor, *args, **kwargs):
        r"""
        Generate an image

        :param torch.Tensor z: latent vector
        """
        return self.model(z, *args, **kwargs)


class CSGMGenerator(Reconstructor):
    r"""CSGMGenerator(backbone_generator=DCGANGenerator(), inf_max_iter=2500, inf_tol=1e-4, inf_lr=1e-2, inf_progress_bar=False)
    Adapts a generator model backbone (e.g DCGAN) for CSGM or AmbientGAN.

    This approach was proposed by :footcite:t:`bora2017compressed` and :footcite:t:`bora2018ambientgan`.

    At train time, the generator samples latent vector from Unif[-1, 1] and passes through backbone.

    At test time, CSGM/AmbientGAN runs an optimisation to find the best latent vector that fits the input
    measurements y, then outputs the corresponding reconstruction.

    This generator can be overridden for more advanced optimisation algorithms by overriding ``optimize_z``.

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for how to use this for adversarial training.

    .. note::

        At train time, this generator discards the measurements ``y``, but these measurements are used at test time.
        This means that train PSNR will be meaningless but test PSNR will be correct.


    :param torch.nn.Module backbone_generator: any neural network that maps a latent vector of length ``nz`` to an image, must have ``nz`` attribute. Defaults to DCGANGenerator()
    :param int inf_max_iter: maximum iterations at inference-time optimisation, defaults to 2500
    :param float inf_tol: tolerance of inference-time optimisation, defaults to 1e-2
    :param float inf_lr: learning rate of inference-time optimisation, defaults to 1e-2
    :param bool inf_progress_bar: whether to display progress bar for inference-time optimisation, defaults to False
    """

    def __init__(
        self,
        backbone_generator: nn.Module | None = None,
        inf_max_iter: int = 2500,
        inf_tol: float = 1e-4,
        inf_lr: float = 1e-2,
        inf_progress_bar: bool = False,
    ):
        if backbone_generator is None:
            backbone_generator = DCGANGenerator()
        super().__init__()
        self.backbone_generator = backbone_generator
        self.inf_loss = MCLoss()
        self.inf_max_iter = inf_max_iter
        self.inf_tol = inf_tol
        self.inf_lr = inf_lr
        self.inf_progress_bar = inf_progress_bar

    def random_latent(self, device, requires_grad=True):
        r"""Generate a latent sample to feed into generative model.

        The model must have an attribute `nz` which is the latent dimension.

        :param torch.device device: torch device
        :param bool requires_grad: whether to require gradient, defaults to True.
        """
        return (
            rand(
                1,
                self.backbone_generator.nz,
                1,
                1,
                device=device,
                requires_grad=requires_grad,
            )
            * 2
            - 1
        )

    def optimize_z(self, z: Tensor, y: Tensor, physics: Physics):
        r"""Run inference-time optimisation of latent z that is consistent with input measurement y according to physics.

        The optimisation is defined with simple stopping criteria. Override this function for more advanced optimisation.

        :param torch.Tensor z: initial latent variable guess
        :param torch.Tensor y: measurement with which to compare reconstructed image
        :param Physics physics: forward model
        :return: optimized latent z
        """
        z = nn.Parameter(z)
        optimizer = Adam([z], lr=self.inf_lr)
        err_prev = 999

        pbar = tqdm(range(self.inf_max_iter), disable=(not self.inf_progress_bar))
        for i in pbar:
            x_hat = self.backbone_generator(z)
            error = self.inf_loss(y=y, x_net=x_hat, physics=physics)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()

            err_curr = error.item()
            err_perc = abs(err_curr - err_prev) / err_curr
            err_prev = err_curr
            pbar.set_postfix({"err_curr": err_curr, "err_perc": err_perc})

            if err_perc < self.inf_tol:
                break
        return z

    def forward(self, y: Tensor, physics: Physics, *args, **kwargs):
        r"""Forward pass of generator model.

        At train time, the generator samples latent vector from Unif[-1, 1] and passes through backbone.

        At test time, CSGM/AmbientGAN runs an optimisation to find the best latent vector that fits the input
        measurements y, then outputs the corresponding reconstruction.

        :param y: measurement to reconstruct
        :param deepinv.physics.Physics physics: forward model
        """
        z = self.random_latent(y.device)

        if not self.training:
            z = self.optimize_z(z, y, physics)

        return self.backbone_generator(z)
