import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import torch
from torch import nn, optim
from torch import autograd
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader,Subset
import torch.nn.utils.spectral_norm as SN

from PIL import Image,ImageOps,ImageEnhance

import cv2
import albumentations as A
from albumentations.pytorch import ToTensor

import glob
import xml.etree.ElementTree as ET #for parsing XML
import shutil
from tqdm import tqdm
import time
import random

from sklearn.metrics import accuracy_score

import torch.backends.cudnn as cudnn

import sys
from evaluation_script.client.mifid_demo import MIFID
from glob import glob

from numpy.random import choice
import random

import pytz
from datetime import datetime
tz = pytz.timezone('Asia/Saigon')


# set params
MODEL_NAME = 'dc_style_v1_5'
LOG = 'log_{}.txt'.format(MODEL_NAME)

LIMIT_DATA = -1

EPOCHS = 500
BATCH_SIZE  = 32
NUM_WORKERS = 4

NC = 3
NZ = 128
NGF = 32
NDF = 40

LR_G = 0.0001
LR_D = 0.0003

BETA1 = 0.5
BETA2 = 0.999

SPECTRAL_NORM = True
NORMALIZATION = 'adain' # selfmod or adain
RANDOM_NOISE = True
USE_STYLE = True

LOSS = 'HINGE' #NS or WGAN or HINGE
PIXEL_NORM = True

USE_SOFT_NOISY_LABELS = True
INVERT_LABELS = True

IMG_SIZE = 128
MEAN1,MEAN2,MEAN3 = 0.5, 0.5, 0.5
STD1,STD2,STD3 = 0.5, 0.5, 0.5

MANUAL_SEED = None
PATH_MODEL_G = ''
PATH_MODEL_D = ''

DIR_IMAGES_INPUT = '/data/cuong/data/motobike_gen/motobike/'
DIR_IMAGES_OUTPUT = '/data/cuong/result/motobike/{}/'.format(MODEL_NAME)

INTRUDERS = [
            '2019_08_05_05_17_32_B0xS_6hHgXG_66398352_483445189138958_8195470045202604419_n_1568719912383_18787.jpg', #
            '22_honda_20Blade_20_3__1568719132927_7959.jpg', #cannot write mode CMYK as PNG
            '50_1_1547807271_1568719515097_13285.jpg',#cannot write mode CMYK as PNG
            '83_6060897e2b1d5627435b1bec2e5a9ac2_1568719487112_12907.jpg',#cannot write mode CMYK as PNG
            '94_banner_tskt_1568719223567_9195.jpg',#cannot write mode CMYK as PNG
            'Motorel38d6l1smallMotor.jpg', # truncated
            'MotorbausxbbzsmallMotor.jpg', # high ratio
            'Motorytec9gywsmallMotor.jpg', # high ratio
            'Motortq4lbb5wsmallMotor.jpg', # outlier
            'Motorjp975mnnsmallMotor.jpg', # outlier
            'Motor_ho4pcmksmallMotor.jpg', # outlier
            'Motor2fankuyqsmallMotor.jpg', # outlier
            'Motorgk66yavfsmallMotor.jpg', # outlier
            ]


def clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def printBoth(filename, args):
    date_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S ')

    # write log
    fo = open(filename, "a")
    fo.write(date_time + args+'\n')
    fo.close()

    # print
    print(date_time + args)

class MotobikeDataset(Dataset):
    def __init__(self, path, img_list, transform1=None, transform2=None):
        self.path      = path
        self.img_list  = img_list
        self.transform1 = transform1
        self.transform2 = transform2

        self.imgs   = []
        self.labels = []
        for i,img_name in enumerate(self.img_list):
            # load image
            img_path = os.path.join(self.path, img_name)
            img = Image.open(img_path).convert('RGB')

            # apply transform
            if self.transform1:
                img = self.transform1(img) #output shape=(ch,h,w)
            if self.transform2:
                img = self.transform2(img)
            self.imgs.append(img)

            #label
            label = 0 #breed_map_2[img_path.split('_')[0]]
            self.labels.append(label)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return {'img':img, 'label':label}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class BatchNormModulate2d(nn.Module):
    """
    Similar to batch norm, but with learnable weights and bias
    """
    def __init__(self, num_features, dim_in, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True, use_sn=True):
        super().__init__()
        self.num_features = num_features
        self.dim_in = dim_in
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Sequential(
            nn.Linear(dim_in, num_features, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(num_features, num_features, bias=False)
        )
        self.beta = nn.Sequential(
            nn.Linear(dim_in, num_features, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(num_features, num_features, bias=False)
        )

    def forward(self, x, z):
        out = self.bn(x)
        gamma = self.gamma(z)
        beta = self.beta(z)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1

class GaussianNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)

class AdaIn(nn.Module):
    """
    latent_dim represents dimension of latent vector similar to style vector in StyleGAN
    """
    def __init__(self, in_channel, latent_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(latent_dim, in_channel * 2)

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class Projection(nn.Module):
    def __init__(self,
                 nz,
                 in_channel,
                 out_channel,
                 shape,
                 bias=False,
                 spectral_norm=False,
                 normalization='selfmod',
                 random_noise=False,
                 use_style=False,
                 use_pixel_norm=False):
        super().__init__()
        self.shape = shape
        self.linear = nn.Linear(in_channel, out_channel, bias=bias)
        self.conv = nn.Conv2d(shape[0], shape[0], 3, 1, 1, bias=bias)
        if spectral_norm:
            self.linear = SN(self.linear)
            self.conv = SN(self.conv)

        self.noise1 = None
        if random_noise:
            self.noise1 = GaussianNoise(shape[0])
            self.noise2 = GaussianNoise(shape[0])

        self.pixel_norm = None
        if use_pixel_norm:
            self.pixel_norm = PixelNorm()

        self.style1 = None
        self.style2 = None

        if normalization == 'adain':
            self.norm1 = AdaIn(shape[0], nz)
            self.norm2 = AdaIn(shape[0], nz)
        else:
            self.norm1 = BatchNormModulate2d(shape[0], nz)
            self.norm2 = BatchNormModulate2d(shape[0], nz)


    def forward(self, x, nz):
        x = self.linear(x)
        x = x.view([x.shape[0]] + self.shape)
        if self.noise1 is not None:
            x = self.noise1(x)
        x = F.leaky_relu(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        x = self.norm1(x, nz)
        x = self.conv(x)
        if self.noise2 is not None:
            x = self.noise2(x)
        x = F.leaky_relu(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        x = self.norm2(x, nz)
        return x

class UpConvBlock(nn.Module):
    """
    normalization is 'selfmod', 'adain'
    """
    def __init__(self,
                 nz,
                 in_channel,
                 out_channel,
                 kernel=4,
                 stride=2,
                 padding=1,
                 bias=False,
                 spectral_norm=False,
                 normalization='selfmod',
                 random_noise=False,
                 use_style=False,
                 use_pixel_norm=False):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=bias)
        if spectral_norm:
            self.conv1 = SN(self.conv1)
            self.conv2 = SN(self.conv2)

        self.noise1 = None
        self.noise2 = None
        if random_noise:
            self.noise1 = GaussianNoise(out_channel)
            self.noise2 = GaussianNoise(out_channel)

        self.pixel_norm = None
        if use_pixel_norm:
            self.pixel_norm = PixelNorm()

        self.style1 = None
        self.style2 = None

        if normalization == 'adain':
            self.norm1 = AdaIn(out_channel, nz)
            self.norm2 = AdaIn(out_channel, nz)
        else:
            self.norm1 = BatchNormModulate2d(out_channel, nz)
            self.norm2 = BatchNormModulate2d(out_channel, nz)


    def forward(self, x, latent):
        x = self.conv1(x)
        if self.noise1 is not None:
            x = self.noise1(x)
        x = F.leaky_relu(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        x = self.norm1(x, latent)
        x = self.conv2(x)
        if self.noise2 is not None:
            x = self.noise2(x)
        x = F.leaky_relu(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        x = self.norm2(x, latent)
        return x


class Generator(nn.Module):
    def __init__(self,
                 nz,
                 nfeats,
                 nchannels,
                 bias=False,
                 spectral_norm=False,
                 normalization='selfmod',
                 random_noise=False,
                 use_style=False,
                 use_pixel_norm=False):
        super(Generator, self).__init__()

        self.mapping = nn.Sequential(
            SN(nn.Linear(nz, nz, bias=bias)),
            nn.LeakyReLU()
        )

        self.linear = Projection(nz, nz, 8*8*nfeats*16, [nfeats*16, 8, 8], bias,
            spectral_norm, normalization, random_noise, use_style, use_pixel_norm) #(nfeats*16) x 8 x 8

        self.conv1 = UpConvBlock(nz, nfeats*16, nfeats*8, 4, 2, 1, bias,
            spectral_norm, normalization, random_noise, use_style, use_pixel_norm) #(nfeats*8) x 16 x 16

        self.conv2 = UpConvBlock(nz, nfeats*8, nfeats*4, 4, 2, 1, bias,
            spectral_norm, normalization, random_noise, use_style, use_pixel_norm) #(nfeats*4) x 32 x 32

        self.conv3 = UpConvBlock(nz, nfeats*4, nfeats*2, 4, 2, 1, bias,
            spectral_norm, normalization, random_noise, use_style, use_pixel_norm) #(nfeats*2) x 64 x 64

        self.conv4 = UpConvBlock(nz, nfeats*2, nfeats*1, 4, 2, 1, bias,
            spectral_norm, normalization, random_noise, use_style, use_pixel_norm) #(nfeats*1) x 128 x 128

        self.conv5 = nn.Conv2d(nfeats*1, nchannels, 1, 1, 0, bias=bias) #(nchannels) x 128 x 128
        if spectral_norm:
            self.conv5 = SN(self.conv5)

    def forward(self, x):
        latent = self.mapping(x)

        out = self.linear(x, latent)

        out = self.conv1(out, latent)

        out = self.conv2(out, latent)

        out = self.conv3(out, latent)

        out = self.conv4(out, latent)

        out = torch.tanh(self.conv5(out))

        return out


class Discriminator(nn.Module):
    def __init__(self, nchannels, nfeats):
        super(Discriminator, self).__init__()

        # input is (nchannels) x 128 x 128
        self.from_rgb = nn.Sequential(
            SN(nn.Conv2d(nchannels, nfeats*1, 1, 1, 0, bias=False)),
            nn.BatchNorm2d(nfeats*1),
            nn.LeakyReLU(0.2)
        )

        self.conv1 = nn.Sequential(
             SN(nn.Conv2d(nfeats*1, nfeats*1, 3, 1, 1, bias=True)),
             nn.BatchNorm2d(nfeats*1),
             nn.LeakyReLU(0.2),
             SN(nn.Conv2d(nfeats*1, nfeats*2, 4, 2, 1, bias=True)),
             nn.BatchNorm2d(nfeats*2),
             nn.LeakyReLU(0.2)
         )
        # (2*nfeats) x 64 x 64

        self.conv2 = nn.Sequential(
            SN(nn.Conv2d(nfeats*2, nfeats*2, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(nfeats*2),
            nn.LeakyReLU(0.2),
            SN(nn.Conv2d(nfeats*2, nfeats*4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(nfeats*4),
            nn.LeakyReLU(0.2)
        )
        # (4*nfeats) x 32 x 32

        self.conv3 = nn.Sequential(
            SN(nn.Conv2d(nfeats*4, nfeats*4, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(nfeats*4),
            nn.LeakyReLU(0.2),
            SN(nn.Conv2d(nfeats*4, nfeats*8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(nfeats*8),
            nn.LeakyReLU(0.2)
        )
        # (8*nfeats) x 16 x 16

        self.conv4 = nn.Sequential(
            SN(nn.Conv2d(nfeats*8, nfeats*8, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(nfeats*8),
            nn.LeakyReLU(0.2),
            SN(nn.Conv2d(nfeats*8, nfeats*16, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(nfeats*16),
            nn.LeakyReLU(0.2)
        )
        # (16*nfeats) x 8 x 8

        self.conv5 = nn.Sequential(
            SN(nn.Conv2d(nfeats*16, nfeats*16, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(nfeats*16),
            nn.LeakyReLU(0.2)
        )
        # (16*nfeats) x 8 x 8

        self.linear = SN(nn.Linear(8*8*nfeats*16, 1, bias=False))


    def forward(self, x):
        x = self.from_rgb(x)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)

        x = x.view(x.shape[0], -1)
        x = self.linear(x)

        return x

def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).to(x.device)
    z = x + alpha * (y - x)

    # gradient penalty
    z = Variable(z, requires_grad=True).to(x.device)
    o = f(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).to(z.device), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()
    return gp

def R1Penalty(real_img, f):
    # gradient penalty
    reals = Variable(real_img, requires_grad=True).to(real_img.device)
    real_logit = f(reals)
    apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
    undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

    real_logit = apply_loss_scaling(torch.sum(real_logit))
    real_grads = grad(real_logit, reals, grad_outputs=torch.ones(real_logit.size()).to(reals.device), create_graph=True)[0].view(reals.size(0), -1)
    real_grads = undo_loss_scaling(real_grads)
    r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
    return r1_penalty

def G_wgan(G, D, nz, batch_size):
    noise = torch.randn(batch_size, nz, device=device)
    fake_images = G(noise)
    fake_logit = D(fake_images)
    G_loss = -fake_logit.mean()
    return G_loss

def D_wgan_gp(G, D, real_images, nz, lammy=10.0, eps=0.001):
    batch_size = real_images.shape[0]
    real_logit = D(real_images)
    noise = torch.randn(batch_size, nz, device=device)
    fake_images = G(noise)
    fake_logit = D(fake_images.detach())
    D_loss = fake_logit.mean() - real_logit.mean()
    D_loss += gradient_penalty(real_images.data, fake_images.data, D) * lammy
#     D_loss += real_logit.mean()**2 * eps
    return D_loss

def D_NS(G, D, real_images, nz, real_labels, fake_labels):
    batch_size = real_images.shape[0]
    real_logit = D(real_images)
    D_loss_real = F.binary_cross_entropy_with_logits(real_logit, real_labels)
    noise = torch.randn(batch_size, nz, device=device)
    fake_images = G(noise)
    fake_logit = D(fake_images.detach())
    D_loss_fake = F.binary_cross_entropy_with_logits(fake_logit, fake_labels)
    D_loss = D_loss_real + D_loss_fake
    return D_loss, D_loss_real.item(), D_loss_fake.item()

def G_NS(G, D, nz, batch_size, real_labels):
    noise = torch.randn(batch_size, nz, device=device)
    fake_images = G(noise)
    fake_logit = D(fake_images)
    G_loss = F.binary_cross_entropy_with_logits(fake_logit, real_labels)
    return G_loss

def D_Hinge(G, D, real_images, nz):
    batch_size = real_images.shape[0]
    real_logit = D(real_images)
    D_loss_real = torch.mean(F.relu(1.0 - real_logit))
    noise = torch.randn(batch_size, nz, device=device)
    fake_images = G(noise)
    fake_logit = D(fake_images.detach())
    D_loss_fake = torch.mean(F.relu(1.0 + fake_logit))
    D_loss = D_loss_real + D_loss_fake
    return D_loss, D_loss_real, D_loss_fake

def G_Hinge(G, D, nz, batch_size):
    noise = torch.randn(batch_size, nz, device=device)
    fake_images = G(noise)
    fake_logit = D(fake_images)
    G_loss = -torch.mean(fake_logit)
    return G_loss


def validate_images_gen(netG, fixed_noise, dir_output):
    gen_images = netG(fixed_noise).to('cpu').clone().detach().squeeze(0)
    gen_images = gen_images*0.5 + 0.5
    for i in range(gen_images.size(0)):
        save_image(gen_images[i, :, :, :], os.path.join(dir_output, '{}.png'.format(i)))

def evaluate_dataset(dir_dataset, mifid):
    img_paths = glob(os.path.join(dir_dataset,'*.*'))

    img_np = np.empty((len(img_paths), 128, 128, 3), dtype=np.uint8)
    for idx, path in tqdm(enumerate(img_paths)):
        img_arr = cv2.imread(path)[..., ::-1]
        img_arr = np.array(img_arr)
        img_np[idx] = img_arr

    score = mifid.compute_mifid(img_np)

    return score

def get_accuracy(output, label):
    output = output.to('cpu').clone().detach().squeeze().numpy()
    output = (output > 0.5).astype('uint8')

    label = label.to('cpu').clone().detach().squeeze().numpy()
    label = (label > 0.5).astype('uint8')

    acc = accuracy_score(output, label)

    return acc

class Trainer:
    def __init__(self, nz, G, D, r1_gamma=0.0, track_grads=False):
        self.nz = nz
        self.track_grads=track_grads
        self.G = G
        self.D = D
        self.fixed_noise = torch.randn(64, self.nz)
        self.r1_gamma = r1_gamma

        self.d_losses = []
        self.g_losses = []
        self.d_losses_real = []
        self.d_losses_fake = []
        self.img_list = []
        self.g_grads = []
        self.d_grads = []

    def check_grads(self, model):
        grads = []
        for n, p in model.named_parameters():
            if not p.grad is None and p.requires_grad and "bias" not in n:
                grads.append(float(p.grad.abs().mean()))
        return grads

    def train(self, epochs, loader, criterion, optim_G, optim_D, scheduler_D, scheduler_G, loss='NS'):
        step = 0
        fixed_noise = torch.randn(128, NZ, device=device)

        for epoch in tqdm(range(epochs)):
            for ii, real_images in enumerate(loader):
                real_images = real_images['img']
                batch_size = real_images.size(0)
                if USE_SOFT_NOISY_LABELS:
                    real_labels = torch.empty((batch_size, 1), device=device).uniform_(0.80, 0.95)
                    fake_labels = torch.empty((batch_size, 1), device=device).uniform_(0.05, 0.20)
                else:
                    real_labels = torch.full((batch_size, 1), 0.95, device=device)
                    fake_labels = torch.full((batch_size, 1), 0.05, device=device)

                if INVERT_LABELS and random.random() < 0.01:
                    real_labels, fake_labels = fake_labels, real_labels
                # Train Discriminator
                self.D.zero_grad()
                real_images = real_images.to(device)

                if loss == 'WGAN':
                    D_loss = D_wgan_gp(self.G, self.D, real_images, self.nz)
                elif loss == 'HINGE':
                    D_loss, D_loss_real, D_loss_fake = D_Hinge(self.G, self.D, real_images, self.nz)
                else:
                    D_loss, D_loss_real, D_loss_fake = D_NS(self.G, self.D, real_images, self.nz, real_labels, fake_labels)

                D_loss.backward()
                optim_D.step()

                # Train Generator
                self.G.zero_grad()

                if loss == 'WGAN':
                    G_loss = G_wgan(self.G, self.D, self.nz, batch_size)
                elif loss == 'HINGE':
                    G_loss = G_Hinge(self.G, self.D, self.nz, batch_size)
                else:
                    G_loss = G_NS(self.G, self.D, self.nz, batch_size, real_labels)
                G_loss.backward()
                optim_G.step()

                step += 1


            # save model
            torch.save(self.G.state_dict(), DIR_IMAGES_OUTPUT + '{}_G.pth'.format(epoch))
            torch.save(self.D.state_dict(), DIR_IMAGES_OUTPUT + '{}_D.pth'.format(epoch))

            # evaluate and save generated images
            with torch.no_grad():
                dir_output = DIR_IMAGES_OUTPUT + str(epoch)
                clean_dir(dir_output)

                validate_images_gen(self.G, fixed_noise, dir_output)
                fdi = evaluate_dataset(dir_output, mifid)

            # print
            printBoth(LOG, 'epoch={}; loss_d={:0.5}; loss_g={:0.5}; fdi={:0.5}'.\
                                        format(epoch, D_loss.item(), G_loss.item(), fdi))

#             scheduler_D.step()
#             scheduler_G.step()

def weights_init(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.kaiming_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
#             m.bias.data.fill_(0.01)

def generate_seed(manualSeed=None):
    if manualSeed is None:
        manualSeed = random.randint(1000, 10000)  # fix seed
    printBoth(LOG, 'RANDOM SEED: {}'.format(manualSeed))
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True

def print_params():
    printBoth(LOG, 'MODEL_NAME = {}'.format(MODEL_NAME))
    printBoth(LOG, 'LOG = {}'.format(LOG))

    printBoth(LOG, 'LIMIT_DATA = {}'.format(LIMIT_DATA))

    printBoth(LOG, 'EPOCHS = {}'.format(EPOCHS))
    printBoth(LOG, 'BATCH_SIZE = {}'.format(BATCH_SIZE))
    printBoth(LOG, 'NUM_WORKERS = {}'.format(NUM_WORKERS))

    printBoth(LOG, 'NC = {}'.format(NC))
    printBoth(LOG, 'NZ = {}'.format(NZ))
    printBoth(LOG, 'NGF = {}'.format(NGF))
    printBoth(LOG, 'NDF = {}'.format(NDF))

    printBoth(LOG, 'LR_G = {}'.format(LR_G))
    printBoth(LOG, 'LR_D = {}'.format(LR_D))

    printBoth(LOG, 'BETA1 = {}'.format(BETA1))
    printBoth(LOG, 'BETA2 = {}'.format(BETA2))

    printBoth(LOG, 'SPECTRAL_NORM = {}'.format(SPECTRAL_NORM))
    printBoth(LOG, 'NORMALIZATION = {}'.format(NORMALIZATION))
    printBoth(LOG, 'RANDOM_NOISE = {}'.format(RANDOM_NOISE))
    printBoth(LOG, 'USE_STYLE = {}'.format(USE_STYLE))

    printBoth(LOG, 'LOSS = {}'.format(LOSS))
    printBoth(LOG, 'PIXEL_NORM = {}'.format(PIXEL_NORM))

    printBoth(LOG, 'USE_SOFT_NOISY_LABELS = {}'.format(USE_SOFT_NOISY_LABELS))
    printBoth(LOG, 'INVERT_LABELS = {}'.format(INVERT_LABELS))

    printBoth(LOG, 'MANUAL_SEED = {}'.format(MANUAL_SEED))
    printBoth(LOG, 'PATH_MODEL_G = {}'.format(PATH_MODEL_G))
    printBoth(LOG, 'PATH_MODEL_D = {}'.format(PATH_MODEL_D))

    printBoth(LOG, 'IMG_SIZE = {}'.format(IMG_SIZE))
    printBoth(LOG, 'MEAN1 = {}; MEAN2 = {}; MEAN3 = {}'.format(MEAN1, MEAN2, MEAN3))
    printBoth(LOG, 'STD1 = {}; STD2 = {}; STD3 = {};'.format(STD1, STD2, STD3))

    printBoth(LOG, 'DIR_IMAGES_INPUT = {}'.format(DIR_IMAGES_INPUT))
    printBoth(LOG, 'DIR_IMAGES_OUTPUT = {}'.format(DIR_IMAGES_OUTPUT))

    printBoth(LOG, 'NUM_WORKERS = {}'.format(NUM_WORKERS))


def generate_images(model_path, dir_images_output, num_images=10000, batch_size=1000, truncated=None, device='cuda'):
    # load model
    netG = Generator(NZ, NGF, 3, False, SPECTRAL_NORM, NORMALIZATION, RANDOM_NOISE, USE_STYLE, PIXEL_NORM).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # generate
    clean_dir(dir_images_output)
    for batch in range(int(num_images/batch_size)):
        #print('Generating batch {}'.format(batch))
        if truncated is not None:
            cont = True
            while cont:
                z = np.random.randn(100*batch_size*NZ)
                z = z[np.where(abs(z)<truncated)]
                if len(z)>=batch_size*NZ:
                    cont = False

            z = torch.from_numpy(z[:batch_size*NZ]).view(batch_size, NZ)
            z = z.float().to(device)
        else:
            z = torch.randn(batch_size, NZ, device=device)

        with torch.no_grad():
            gen_images = netG(z)
        gen_images = gen_images.to('cpu').clone().detach()
        gen_images = gen_images*0.5 + 0.5

        for i in range(gen_images.size(0)):
            save_image(gen_images[i, :, :, :], os.path.join(dir_images_output, '{}_{}.png'.format(batch, i)))

if __name__ == '__main__':
    # load the evaluation model
    printBoth(LOG, 'Loading the evaluation model ...')
    mifid = MIFID(model_path='./evaluation_script/client/motorbike_classification_inception_net_128_v4_e36.pb',
                  public_feature_path='./evaluation_script/client/public_feature.npz')

    # set seeds
    generate_seed(MANUAL_SEED)

    # params
    print_params()

    # create transform
    printBoth(LOG, 'Creating dataloaders ...')
    transform1 = transforms.Compose([transforms.Resize(IMG_SIZE)])

    transform2 = transforms.Compose([transforms.RandomCrop(IMG_SIZE),
                                     transforms.ColorJitter(),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[MEAN1, MEAN2, MEAN3],
                                                          std=[STD1, STD2, STD3]),
                                    ])

    img_filenames = []
    for image_name in sorted(os.listdir(DIR_IMAGES_INPUT)):
        if image_name not in INTRUDERS:
            img_filenames.append(image_name)

        if (LIMIT_DATA>0) and (len(img_filenames)>=LIMIT_DATA):
            break
    printBoth(LOG, 'The length of img_filenames = {}'.format(len(img_filenames)))

    # create dataloader
    train_set = MotobikeDataset(path=DIR_IMAGES_INPUT,
                                img_list=img_filenames,
                                transform1=transform1,
                                transform2=transform2,
                          )
    train_loader = DataLoader(train_set,
                              shuffle=True,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=True)
    printBoth(LOG, 'The length of train_set = {}'.format(len(train_set)))
    printBoth(LOG, 'The length of train_loader = {}'.format(len(train_loader)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    printBoth(LOG, 'DEVICE = {}'.format(device))

    # train
    netG = Generator(NZ, NGF, 3, False, SPECTRAL_NORM, NORMALIZATION, RANDOM_NOISE, USE_STYLE, PIXEL_NORM).to(device)
    netD = Discriminator(3, NDF).to(device)

    if PATH_MODEL_G is not '':
        netG.load_state_dict(torch.load(PATH_MODEL_G, map_location=torch.device(device)))
    if PATH_MODEL_D is not '':
        netD.load_state_dict(torch.load(PATH_MODEL_D, map_location=torch.device(device)))

    printBoth(LOG, 'count_parameters of netG = {}'.format(count_parameters(netG)))
    printBoth(LOG, 'count_parameters of netD = {}'.format(count_parameters(netD)))

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(BETA1, BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    scheduler_D = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    scheduler_G = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)

    # train
    clean_dir(DIR_IMAGES_OUTPUT)
    trainer = Trainer(NZ, netG, netD, track_grads=True)
    trainer.train(EPOCHS, train_loader, criterion, optimizerG, optimizerD, scheduler_D, scheduler_G, loss=LOSS)
