import os
import random, math
from math import sqrt
import time
import copy
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn.utils import spectral_norm
from torch.autograd import Function, Variable, grad

from torchvision import datasets, transforms, utils
from torchvision.utils import save_image, make_grid

from scipy.stats import truncnorm

import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
import shutil

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from scipy.stats import truncnorm

import sys
import cv2
from evaluation_script.client.mifid_demo import MIFID
from glob import glob

import gc
gc.enable()

import pytz
from datetime import datetime
tz = pytz.timezone('Asia/Saigon')


# config
MODEL_NAME = 'style_v1'
LOG = 'log_{}.txt'.format(MODEL_NAME)

LIMIT_DATA = -1

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
MAX_ITERATIONS = 3_000_000
STATE_PRINT_ITERS = 100

CODE_SIZE = 128
N_MLP = 8
N_CRITIC = 1

class Args:
    base_lr = 0.0015
    lr = {8: 0.002, 16: 0.004, 32: 0.006, 64: 0.008, 128: 0.008} # markpeng - faster learing rate
    batch = {8: 128, 16: 64,  32: 32,  64: 32,  128: 32}
    phase = {8: 400_000, 16: 400_000, 32: 400_000, 64: 400_000, 128: 1_000_000}
    #phase = {8: 4_000, 16: 4_000, 32: 4_000, 64: 4_000, 128: 4_000}
    init_size = 8
    max_size = 128
    mixing = True
    loss = 'r1'  # or 'wgan-gp'

args = Args()

BETA1 = 0.0
BETA2 = 0.99



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


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True


def load_dataset_images(root, n_samples=25000, image_size=args.max_size):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                      '.tiff', '.webp')

    def is_valid_file(x):
        return datasets.folder.has_file_allowed_extension(x, IMG_EXTENSIONS)

    required_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])

    imgs = []
    paths = []
    for root, _, fnames in sorted(os.walk(root)):
        for fname in sorted(fnames)[:min(n_samples, 999999999999999)]:
            if fname not in INTRUDERS:
                path = os.path.join(root, fname)
                paths.append(path)

            if (LIMIT_DATA>0) and (len(paths)>=LIMIT_DATA):
                break

    for path in paths:
        if is_valid_file(path):
            img = datasets.folder.default_loader(path)
            img = required_transforms(img)
            imgs.append(img)

    return imgs


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_images=None,
                 transform=None,
                 target_image_size=args.max_size):

        self.dataset_images = dataset_images
        self.transform = transform
        self.target_image_size = target_image_size
        self.samples = []

        if self.target_image_size < args.max_size:
            required_transforms = transforms.Compose([transforms.Resize(self.target_image_size)])
            for img in dataset_images:
                self.samples.append(required_transforms(img))
        else:
            self.samples = self.dataset_images

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return np.asarray(sample)

    def __len__(self):
        return len(self.samples)

def sample_data(batch_size, images, image_size=8):
    train_data = DataGenerator(images,
                               transform=transform,
                               target_image_size=image_size)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               shuffle=True,
                                               batch_size=BATCH_SIZE,
                                               num_workers=4)

    return train_loader


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] +
                  weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1]) / 4

        out = F.conv_transpose2d(input,
                               weight,
                               self.bias,
                               stride=2,
                               padding=self.pad)

        return out


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] +
                  weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1]) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(
            torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(grad_output,
                              kernel_flip,
                              padding=1,
                              groups=grad_output.shape[1])

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(gradgrad_output,
                              kernel,
                              padding=1,
                              groups=gradgrad_output.shape[1])

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel,
                                                kernel_flip)

        return grad_input, None, None


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                              dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip',
                             weight_flip.repeat(channel, 1, 1, 1))
        self.blur = BlurFunction.apply

    def forward(self, input):
        return self.blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        conv = conv
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            padding,
            kernel_size2=None,
            padding2=None,
            downsample=False,
            fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel,
                                    out_channel,
                                    kernel2,
                                    padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel,
                                out_channel,
                                kernel2,
                                padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:
                                    in_channel] = 1  # set bias to 1 for style associated
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            padding=1,
            style_dim=CODE_SIZE,
            initial=False,
            upsample=False,
            fused=False,
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(in_channel,
                                      out_channel,
                                      kernel_size,
                                      padding=padding),
                        Blur(out_channel),
                    )

                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(in_channel,
                                    out_channel,
                                    kernel_size,
                                    padding=padding),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = EqualConv2d(in_channel,
                                         out_channel,
                                         kernel_size,
                                         padding=padding)

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel,
                                 out_channel,
                                 kernel_size,
                                 padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out


class Generator(nn.Module):
    def __init__(self, code_dim, fused=True):
        super().__init__()

        self.progression = nn.ModuleList([
            StyledConvBlock(128, 128, 3, 1, initial=True),  # 4
            StyledConvBlock(128, 128, 3, 1, upsample=True),  # 8
            StyledConvBlock(128, 64, 3, 1, upsample=True),  # 16
            StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),  # 32
            StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 64
            StyledConvBlock(16,  8, 3, 1, upsample=True, fused=fused),  # 128
        ])

        self.to_rgb = nn.ModuleList([
            EqualConv2d(128, 3, 1),
            EqualConv2d(128, 3, 1),
            EqualConv2d(64, 3, 1),
            EqualConv2d(32, 3, 1),
            EqualConv2d(16, 3, 1),
            EqualConv2d(8, 3, 1),
        ])

        # self.blur = Blur()

    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = random.sample(list(range(step)), len(style) - 1)

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(
                        inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out

                out = conv(out, style_step, noise[i])

            else:
                out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb,
                                             scale_factor=2,
                                             mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=128, n_mlp=8):
        super().__init__()

        self.generator = Generator(code_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(
            self,
            input,
            noise=None,
            step=0,
            alpha=-1,
            mean_style=None,
            style_weight=0,
            mixing_range=(-1, -1),
    ):
        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2**i
                noise.append(
                    torch.randn(batch, 1, size, size, device=input[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight *
                                   (style - mean_style))

            styles = styles_norm

        return self.generator(styles,
                              noise,
                              step,
                              alpha,
                              mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style


class Discriminator(nn.Module):
    def __init__(self, fused=True):
        super().__init__()

        self.progression = nn.ModuleList([
            ConvBlock( 8, 16, 3, 1, downsample=True, fused=fused),  # 128
            ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 64
            ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 32
            ConvBlock(64, 128, 3, 1, downsample=True),              # 14
            ConvBlock(128, 128, 3, 1, downsample=True),             # 8
            ConvBlock(129, 128, 3, 1, 4, 0),                        # 4
        ])

        self.from_rgb = nn.ModuleList([
            EqualConv2d(3,  8, 1),
            EqualConv2d(3, 16, 1),
            EqualConv2d(3, 32, 1),
            EqualConv2d(3, 64, 1),
            EqualConv2d(3, 128, 1),
            EqualConv2d(3, 128, 1),
        ])

        # self.blur = Blur()
        self.n_layer = len(self.progression)
        self.linear = EqualLinear(CODE_SIZE, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                # Minibatch stddev
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)

        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate_images_gen(netG, z_fixed, step, alpha, dir_output):
    gen_images = netG(z_fixed, step=step, alpha=alpha).to('cpu').clone().detach().squeeze(0)
    gen_images = gen_images*0.5 + 0.5
    for i in range(gen_images.size(0)):
        save_image(gen_images[i, :, :, :], os.path.join(dir_output, '{}.png'.format(i)))

def evaluate_dataset(dir_dataset, mifid, step):
    img_paths = glob(os.path.join(dir_dataset,'*.*'))

    resolution = 4 * 2**step
    img_np = np.empty((len(img_paths), resolution, resolution, 3), dtype=np.uint8)
    for idx, path in tqdm(enumerate(img_paths)):
        img_arr = cv2.imread(path)[..., ::-1]
        img_arr = np.array(img_arr)
        img_np[idx] = img_arr

    score = mifid.compute_mifid(img_np)

    return score

def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


# empty cache
torch.cuda.empty_cache()

# generate seeds
printBoth(LOG, 'Generating seeds ...')
seed_everything()

# load the evaluation model
printBoth(LOG, 'Loading the evaluation model ...')
mifid = MIFID(model_path='./evaluation_script/client/motorbike_classification_inception_net_128_v4_e36.pb',
              public_feature_path='./evaluation_script/client/public_feature.npz')

# create models
printBoth(LOG, 'Generating models ...')
generator = StyledGenerator(CODE_SIZE, N_MLP).to(DEVICE)
discriminator = Discriminator().to(DEVICE)
g_running = StyledGenerator(CODE_SIZE, N_MLP).to(DEVICE)
g_running.train(False)


# create
printBoth(LOG, 'Creating optimizers ...')
g_optimizer = optim.Adam(generator.generator.parameters(),
                         lr=args.base_lr,
                         betas=(BETA1, BETA2))
g_optimizer.add_param_group({
    'params': generator.style.parameters(),
    'lr': args.base_lr * 0.01,
    'mult': 0.01,
})
d_optimizer = optim.Adam(discriminator.parameters(),
                         lr=args.base_lr,
                         betas=(BETA1, BETA2))

accumulate(g_running, generator, 0)

printBoth(LOG, 'Generator params = {}'.format(count_parameters(generator)))
printBoth(LOG, 'Discriminator params = {}'.format(count_parameters(discriminator)))


# load images
printBoth(LOG, 'Loading images ...')
images = load_dataset_images(DIR_IMAGES_INPUT)
printBoth(LOG, 'Number of images = {}'.format(len(images)))

# train
printBoth(LOG, 'Start training ...')
clean_dir(DIR_IMAGES_OUTPUT)

gc.collect()

disc_loss_val = 0
gen_loss_val = 0
grad_loss_val = 0

alpha = 0
used_sample = 0

step = int(math.log2(args.init_size)) - 2
resolution = 4 * 2**step
phase = args.phase.get(resolution)

loader = sample_data(args.batch.get(resolution, BATCH_SIZE), images, image_size=resolution)
data_loader = iter(loader)

max_step = int(math.log2(args.max_size)) - 2  # 5
final_progress = False

adjust_lr(g_optimizer, args.lr.get(resolution, args.base_lr))
adjust_lr(d_optimizer, args.lr.get(resolution, args.base_lr))
requires_grad(generator, False)
requires_grad(discriminator, True)

z_fixed = truncated_normal((128, CODE_SIZE), threshold=1.0)
z_fixed = torch.from_numpy(z_fixed).float().to(DEVICE)

printBoth(LOG, 'resolution = {}x{}'.format(resolution, resolution))
printBoth(LOG, 'step = {}'.format(step))
printBoth(LOG, 'batch_size = {}'.format(args.batch.get(resolution, BATCH_SIZE)))
printBoth(LOG, 'Generator LR = {}'.format(g_optimizer.state_dict()['param_groups'][0]['lr']))
printBoth(LOG, 'Style LR = {}'.format(g_optimizer.state_dict()['param_groups'][1]['lr']))
printBoth(LOG, 'Discriminator LR = {}'.format(d_optimizer.state_dict()['param_groups'][0]['lr']))


for iters in range(MAX_ITERATIONS):
    gc.collect()

    # get phase, resolution, alpha and dataloader
    alpha = min(1, 1 / phase * (used_sample + 1))

    if resolution == args.init_size or final_progress:
        alpha = 1

    if used_sample > phase * 2:
        used_sample = 0
        step += 1

        if step > max_step:
            step = max_step
            final_progress = True
        else:
            alpha = 0

        resolution = 4 * 2**step
        phase = args.phase.get(resolution)

        del loader
        gc.collect()
        loader = sample_data(args.batch.get(resolution, BATCH_SIZE), images, image_size=resolution)
        data_loader = iter(loader)

        adjust_lr(g_optimizer, args.lr.get(resolution, args.base_lr))
        adjust_lr(d_optimizer, args.lr.get(resolution, args.base_lr))

        z_fixed = truncated_normal((128, CODE_SIZE), threshold=1.0)
        z_fixed = torch.from_numpy(z_fixed).float().to(DEVICE)

        printBoth(LOG, 'resolution = {}x{}'.format(resolution, resolution))
        printBoth(LOG, 'step = {}'.format(step))
        printBoth(LOG, 'batch_size = {}'.format(args.batch.get(resolution, BATCH_SIZE)))
        printBoth(LOG, 'Generator LR = {}'.format(g_optimizer.state_dict()['param_groups'][0]['lr']))
        printBoth(LOG, 'Style LR = {}'.format(g_optimizer.state_dict()['param_groups'][1]['lr']))
        printBoth(LOG, 'Discriminator LR = {}'.format(d_optimizer.state_dict()['param_groups'][0]['lr']))

    try:
        real_image = next(data_loader).to(DEVICE)
    except (OSError, StopIteration):
        data_loader = iter(loader)
        real_image = next(data_loader).to(DEVICE)

    used_sample += real_image.shape[0]
    b_size = real_image.size(0)

    # train discriminator on real images
    discriminator.zero_grad()
    if args.loss == 'wgan-gp':
        real_predict = discriminator(real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() - 0.001 * (real_predict**2).mean()
        (-real_predict).backward()

    elif args.loss == 'r1':
        real_image.requires_grad = True
        real_predict = discriminator(real_image, step=step, alpha=alpha)
        real_predict = F.softplus(-real_predict).mean()
        real_predict.backward(retain_graph=True)

        grad_real = grad(outputs=real_predict.sum(),inputs=real_image,create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0),-1).norm(2, dim=1)**2).mean()
        grad_penalty = 10 / 2 * grad_penalty
        grad_penalty.backward()
        grad_loss_val = grad_penalty.item()

    # train discriminator on fake images
    if args.mixing and random.random() < 0.9:
        gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(4, b_size, CODE_SIZE, device=DEVICE).chunk(4, 0)
        gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
        gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]
    else:
        gen_in1, gen_in2 = torch.randn(2, b_size, CODE_SIZE, device=DEVICE).chunk(2, 0)
        gen_in1 = gen_in1.squeeze(0)
        gen_in2 = gen_in2.squeeze(0)

    fake_image = generator(gen_in1, step=step, alpha=alpha)
    fake_predict = discriminator(fake_image, step=step, alpha=alpha)
    if args.loss == 'wgan-gp':
        fake_predict = fake_predict.mean()
        fake_predict.backward()

        eps = torch.rand(b_size, 1, 1, 1).to(DEVICE)
        x_hat = eps * real_image.data + (1 - eps) * fake_image.data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(outputs=hat_predict.sum(),
                          inputs=x_hat,
                          create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) -1)**2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val = grad_penalty.item()
        disc_loss_val = (real_predict - fake_predict).item()

    elif args.loss == 'r1':
        fake_predict = F.softplus(fake_predict).mean()
        fake_predict.backward()
        disc_loss_val = (real_predict + fake_predict).item()

    d_optimizer.step()


    # train generator
    if (iters + 1) % N_CRITIC == 0:
        generator.zero_grad()

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        fake_image = generator(gen_in2, step=step, alpha=alpha)
        predict = discriminator(fake_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            loss = -predict.mean()

        elif args.loss == 'r1':
            loss = F.softplus(-predict).mean()

        gen_loss_val = loss.item()

        loss.backward()
        g_optimizer.step()
        accumulate(g_running, generator)

        requires_grad(generator, False)
        requires_grad(discriminator, True)


    if (iters) % STATE_PRINT_ITERS == 0:
        # save model
        if args.max_size == resolution:
            torch.save(generator.state_dict(),      DIR_IMAGES_OUTPUT + '{}_G.pth'.format(iters))
            torch.save(discriminator.state_dict(),  DIR_IMAGES_OUTPUT + '{}_D.pth'.format(iters))

        # evaluate
        with torch.no_grad():
            dir_output = DIR_IMAGES_OUTPUT + str(iters)
            clean_dir(dir_output)

            fdi = 0
            if args.max_size == resolution:
                validate_images_gen(generator, z_fixed, step, alpha, dir_output)
                fdi = evaluate_dataset(dir_output, mifid, step)

        print('[{}/{}] phase={}; resolution={}; used_samples={}; Grad={:0.4f}; Alpha={:0.4f}; Loss_G={:0.4f}; Loss_D={:0.4f}; fdi={}'.\
              format(iters, MAX_ITERATIONS, phase, 4*2**step, used_sample, grad_loss_val, alpha, gen_loss_val, disc_loss_val, fdi))
