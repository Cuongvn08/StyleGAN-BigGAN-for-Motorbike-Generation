# import
import os
import scipy.misc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
#%matplotlib inline
import numpy as np
import torch
from torch import nn, optim
from torch import autograd
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader,Subset
from PIL import Image,ImageOps,ImageEnhance
import torchvision.utils as vutils

import torch.backends.cudnn as cudnn

import cv2
import albumentations as A
from albumentations.pytorch import ToTensor

import glob
import xml.etree.ElementTree as ET #for parsing XML
import shutil
from tqdm import tqdm
import time
import random

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_

import pytz
from datetime import datetime
tz = pytz.timezone('Asia/Saigon')

import sys
from evaluation_script.client.mifid_demo import MIFID
from glob import glob


manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
cudnn.benchmark = True


class Config():
    LIMIT_DATA = -1

    MODEL_NAME = 'sa_v1'
    LOG = 'log_{}.txt'.format(MODEL_NAME)

    IMAGE_SIZE = 128
    MEAN1,MEAN2,MEAN3 = 0.5, 0.5, 0.5
    STD1,STD2,STD3    = 0.5, 0.5, 0.5

    EPOCHES = 500 # Number of training epochs
    BATCH_SIZE  = 64
    NUM_WORKERS = 4

    NGPU = 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_CLASSES = 1
    NC = 3 # Number of channels in the training images. For color images this is 3
    Z_DIM = 128 # Size of z latent vector (i.e. size of generator input)
    G_CONV_DIM = 32 #64 # Size of feature maps in generator
    D_CONV_DIM = 32 #64 # Size of feature maps in discriminator
    LR_G=0.001
    LR_D=0.001
    BETA1 = 0.0
    BETA2 = 0.999

    REAL_LABEL = 0.9
    FAKE_LABEL = 0.0

    NUM_IMAGES_GEN = 64

    PATH_PRETRAINED_G = ''
    PATH_PRETRAINED_D = ''

    DIR_IMAGES_INPUT = '/data/cuong/data/motobike_gen/motobike/'
    DIR_IMAGES_OUTPUT = '/data/cuong/result/motobike/{}/'.format(MODEL_NAME)

    INTRUDERS = [
            '2019_08_05_05_17_32_B0xS_6hHgXG_66398352_483445189138958_8195470045202604419_n_1568719912383_18787.jpg', #
            '22_honda_20Blade_20_3__1568719132927_7959.jpg', #cannot write mode CMYK as PNG
            '50_1_1547807271_1568719515097_13285.jpg',#cannot write mode CMYK as PNG
            '83_6060897e2b1d5627435b1bec2e5a9ac2_1568719487112_12907.jpg',#cannot write mode CMYK as PNG
            '94_banner_tskt_1568719223567_9195.jpg',#cannot write mode CMYK as PNG
            'Motorel38d6l1smallMotor.jpg', # truncated
        ]


# utilities
def printBoth(filename, args):
    date_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S ')

    # write log
    fo = open(filename, "a")
    fo.write(date_time + args+'\n')
    fo.close()

    # print
    print(date_time + args)

def clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def generate_img(netG,fixed_noise,fixed_aux_labels=None):
    if fixed_aux_labels is not None:
        gen_image = netG(fixed_noise,fixed_aux_labels).to('cpu').clone().detach().squeeze(0)
    else:
        gen_image = netG(fixed_noise).to('cpu').clone().detach().squeeze(0)
    #denormalize
    gen_image = gen_image*0.5 + 0.5
    gen_image_numpy = gen_image.numpy().transpose(0,2,3,1)
    return gen_image_numpy

def show_generate_imgs(netG,fixed_noise,fixed_aux_labels=None):
    gen_images_numpy = generate_img(netG,fixed_noise,fixed_aux_labels)
    fig = plt.figure(figsize=(25, 16))
    # display 10 images from each class
    for i, img in enumerate(gen_images_numpy):
        ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
        plt.imshow(img)
    plt.show()
    plt.close()

def validate_images_gen(netG, fixed_noise, fixed_labels, dir_output):
    gen_images = netG(fixed_noise, fixed_labels).to('cpu').clone().detach().squeeze(0)
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

# dataloader
class MotobikeDataset(Dataset):
    def __init__(self, path, img_list, transform1=None, transform2=None):
        self.path      = path
        self.img_list  = img_list
        self.transform1 = transform1
        self.transform2 = transform2

        self.imgs   = []
        self.labels = []
        for i,img_name in enumerate(self.img_list):
            if img_name in Config.INTRUDERS:
                continue

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

# model
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out


class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].fill_(1.)  # Initialize scale to 1
        self.embed.weight.data[:, num_features:].zero_()  # Initialize bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(GenBlock, self).__init__()
        self.cond_bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, labels):
        x0 = x

        x = self.cond_bn1(x, labels)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.cond_bn2(x, labels)
        x = self.relu(x)
        x = self.snconv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class Generator(nn.Module):
    """Generator."""

    def __init__(self, z_dim, g_conv_dim, num_classes):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.g_conv_dim = g_conv_dim
        self.snlinear0 = snlinear(in_features=z_dim, out_features=g_conv_dim*16*4*4)
        self.block1 = GenBlock(g_conv_dim*16, g_conv_dim*16, num_classes)
        self.block2 = GenBlock(g_conv_dim*16, g_conv_dim*8, num_classes)
        self.block3 = GenBlock(g_conv_dim*8, g_conv_dim*4, num_classes)
        self.self_attn = Self_Attn(g_conv_dim*4)
        self.block4 = GenBlock(g_conv_dim*4, g_conv_dim*2, num_classes)
        self.block5 = GenBlock(g_conv_dim*2, g_conv_dim, num_classes)
        self.bn = nn.BatchNorm2d(g_conv_dim, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=g_conv_dim, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z, labels):
        # n x z_dim
        act0 = self.snlinear0(z)            # n x g_conv_dim*16*4*4
        act0 = act0.view(-1, self.g_conv_dim*16, 4, 4) # n x g_conv_dim*16 x 4 x 4
        act1 = self.block1(act0, labels)    # n x g_conv_dim*16 x 8 x 8
        act2 = self.block2(act1, labels)    # n x g_conv_dim*8 x 16 x 16
        act3 = self.block3(act2, labels)    # n x g_conv_dim*4 x 32 x 32
        act3 = self.self_attn(act3)         # n x g_conv_dim*4 x 32 x 32
        act4 = self.block4(act3, labels)    # n x g_conv_dim*2 x 64 x 64
        act5 = self.block5(act4, labels)    # n x g_conv_dim  x 128 x 128
        act5 = self.bn(act5)                # n x g_conv_dim  x 128 x 128
        act5 = self.relu(act5)              # n x g_conv_dim  x 128 x 128
        act6 = self.snconv2d1(act5)         # n x 3 x 128 x 128
        act6 = self.tanh(act6)              # n x 3 x 128 x 128
        return act6


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, d_conv_dim, num_classes):
        super(Discriminator, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.opt_block1 = DiscOptBlock(3, d_conv_dim)
        self.block1 = DiscBlock(d_conv_dim, d_conv_dim*2)
        self.self_attn = Self_Attn(d_conv_dim*2)
        self.block2 = DiscBlock(d_conv_dim*2, d_conv_dim*4)
        self.block3 = DiscBlock(d_conv_dim*4, d_conv_dim*8)
        self.block4 = DiscBlock(d_conv_dim*8, d_conv_dim*16)
        self.block5 = DiscBlock(d_conv_dim*16, d_conv_dim*16)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=d_conv_dim*16, out_features=1)
        self.sn_embedding1 = sn_embedding(num_classes, d_conv_dim*16)
        self.sigmoid = nn.Sigmoid()

        # Weight init
        self.apply(init_weights)
        xavier_uniform_(self.sn_embedding1.weight)

    def forward(self, x, labels):
        # n x 3 x 128 x 128
        h0 = self.opt_block1(x) # n x d_conv_dim   x 64 x 64
        h1 = self.block1(h0)    # n x d_conv_dim*2 x 32 x 32
        h1 = self.self_attn(h1) # n x d_conv_dim*2 x 32 x 32
        h2 = self.block2(h1)    # n x d_conv_dim*4 x 16 x 16
        h3 = self.block3(h2)    # n x d_conv_dim*8 x  8 x  8
        h4 = self.block4(h3)    # n x d_conv_dim*16 x 4 x  4
        h5 = self.block5(h4, downsample=False)  # n x d_conv_dim*16 x 4 x 4
        h5 = self.relu(h5)              # n x d_conv_dim*16 x 4 x 4
        h6 = torch.sum(h5, dim=[2,3])   # n x d_conv_dim*16
        output1 = torch.squeeze(self.snlinear1(h6)) # n

        # Projection
        h_labels = self.sn_embedding1(labels)   # n x d_conv_dim*16
        proj = torch.mul(h6, h_labels)          # n x d_conv_dim*16
        output2 = torch.sum(proj, dim=[1])      # n
        # Out
        output = output1 + output2              # n
        output = self.sigmoid(output)           # n

        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_params():
    printBoth(Config.LOG, 'MODEL_NAME={}'.format(Config.MODEL_NAME))
    printBoth(Config.LOG, 'LOG={}'.format(Config.LOG))

    printBoth(Config.LOG, 'DEVICE={}'.format(Config.DEVICE))
    printBoth(Config.LOG, 'NGPU={}'.format(Config.NGPU))

    printBoth(Config.LOG, 'IMAGE_SIZE={}'.format(Config.IMAGE_SIZE))
    printBoth(Config.LOG, 'MEAN1={}; MEAN2={}; MEAN3={}'.format(Config.MEAN1, Config.MEAN2, Config.MEAN3))
    printBoth(Config.LOG, 'STD1={}; STD2={}; STD3={};'.format(Config.STD1, Config.STD2, Config.STD3))

    printBoth(Config.LOG, 'BATCH_SIZE={}'.format(Config.BATCH_SIZE))
    printBoth(Config.LOG, 'NUM_WORKERS={}'.format(Config.NUM_WORKERS))
    printBoth(Config.LOG, 'EPOCHES={}'.format(Config.EPOCHES))

    printBoth(Config.LOG, 'LR_G={}'.format(Config.LR_G))
    printBoth(Config.LOG, 'LR_D={}'.format(Config.LR_D))
    printBoth(Config.LOG, 'BETA1={}'.format(Config.BETA1))
    printBoth(Config.LOG, 'BETA2={}'.format(Config.BETA2))

    printBoth(Config.LOG, 'NUM_CLASSES={}'.format(Config.NUM_CLASSES))
    printBoth(Config.LOG, 'NC={}'.format(Config.NC))
    printBoth(Config.LOG, 'Z_DIM={}'.format(Config.Z_DIM))
    printBoth(Config.LOG, 'G_CONV_DIM={}'.format(Config.G_CONV_DIM))
    printBoth(Config.LOG, 'D_CONV_DIM={}'.format(Config.D_CONV_DIM))

    printBoth(Config.LOG, 'REAL_LABEL={}'.format(Config.REAL_LABEL))
    printBoth(Config.LOG, 'FAKE_LABEL={}'.format(Config.FAKE_LABEL))

    printBoth(Config.LOG, 'NUM_IMAGES_GEN={}'.format(Config.NUM_IMAGES_GEN))

    printBoth(Config.LOG, 'PATH_PRETRAINED_G={}'.format(Config.PATH_PRETRAINED_G))
    printBoth(Config.LOG, 'PATH_PRETRAINED_D={}'.format(Config.PATH_PRETRAINED_D))

    printBoth(Config.LOG, 'DIR_IMAGES_INPUT={}'.format(Config.DIR_IMAGES_INPUT))
    printBoth(Config.LOG, 'DIR_IMAGES_OUTPUT={}'.format(Config.DIR_IMAGES_OUTPUT))


def create_dataloader():
    printBoth(Config.LOG, 'Creating dataloader ...')

    # parse images
    img_filenames = []
    for image_name in sorted(os.listdir(Config.DIR_IMAGES_INPUT)):
        if image_name not in Config.INTRUDERS:
            img_filenames.append(image_name)

        if (Config.LIMIT_DATA>0) and (len(img_filenames)>Config.LIMIT_DATA):
            break

    # create transform
    transform1 = transforms.Compose([transforms.Resize(Config.IMAGE_SIZE)])
    transform2 = transforms.Compose([transforms.RandomCrop(Config.IMAGE_SIZE),
                                     #transforms.RandomAffine(degrees=5),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     #transforms.RandomApply(random_transforms, p=0.3),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[Config.MEAN1, Config.MEAN2, Config.MEAN3],
                                                          std=[Config.STD1, Config.STD2, Config.STD3]),
                                    ])

    # creat dataloader
    train_set = MotobikeDataset(path=Config.DIR_IMAGES_INPUT,
                               img_list=img_filenames,
                               transform1=transform1,
                               transform2=transform2,
                              )
    train_loader = DataLoader(train_set,
                              shuffle=True,
                              batch_size=Config.BATCH_SIZE,
                              num_workers=Config.NUM_WORKERS)

    printBoth(Config.LOG, 'The length of train_set = {}'.format(len(train_set)))
    printBoth(Config.LOG, 'The length of dataloader = {}'.format(len(train_loader)))

    return train_loader


def create_models():
    printBoth(Config.LOG, 'Creating models ...')

    # Create the generator
    netG = Generator(z_dim=Config.Z_DIM, g_conv_dim=Config.G_CONV_DIM, num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

    if Config.PATH_PRETRAINED_G is not '':
        netG.load_state_dict(torch.load(Config.PATH_PRETRAINED_G, map_location=Config.DEVICE))
    if (Config.DEVICE.type == 'cuda') and (Config.NGPU > 1):
        netG = nn.DataParallel(netG, list(range(Config.NGPU)))

    printBoth(Config.LOG, 'count_params of netG = {}'.format(count_parameters(netG)))

    # Create the Discriminator
    netD = Discriminator(d_conv_dim=Config.D_CONV_DIM, num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

    if Config.PATH_PRETRAINED_D is not '':
        netD.load_state_dict(torch.load(Config.PATH_PRETRAINED_D, map_location=Config.DEVICE))
    if (Config.DEVICE.type == 'cuda') and (Config.NGPU > 1):
        netD = nn.DataParallel(netD, list(range(Config.NGPU)))

    printBoth(Config.LOG, 'count_params of netD = {}'.format(count_parameters(netD)))

    return netG, netD


def train(train_loader, netG, netD, mifid):
    printBoth(Config.LOG, 'Training ...')

    # train
    optimizerG = torch.optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()),
                                   lr=Config.LR_G,
                                   betas=[Config.BETA1, Config.BETA2])
    optimizerD = torch.optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()),
                                   lr=Config.LR_D,
                                   betas=[Config.BETA1, Config.BETA2])

    criterion = nn.BCELoss()

    # noises for generation
    fixed_noise = torch.randn(Config.NUM_IMAGES_GEN, Config.Z_DIM, device=Config.DEVICE)
    fixed_labels = torch.tensor(np.zeros(Config.NUM_IMAGES_GEN)).long().to(Config.DEVICE, non_blocking=True) #shape=(*,)

    # train
    clean_dir(Config.DIR_IMAGES_OUTPUT)

    netG.train()
    netD.train()

    for epoch in range(Config.EPOCHES):
        list_errD = []
        list_errG = []
        for i, data in enumerate(train_loader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real = data['img'].to(Config.DEVICE)
            labels = data['label'].to(Config.DEVICE)
            b_size = real.size(0)
            label = torch.full((b_size,), Config.REAL_LABEL, device=Config.DEVICE)
            # Forward pass real batch through D
            output = netD(real,labels).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            #noise = torch.randn(b_size, z_dim, 1, 1, device=DEVICE)
            noise = torch.randn(b_size, Config.Z_DIM, device=Config.DEVICE)
            # Generate fake image batch with G
            fake = netG(noise,labels)
            label.fill_(Config.FAKE_LABEL)
            # Classify all fake batch with D
            output = netD(fake.detach(),labels).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            list_errD.append(errD.item())

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(Config.REAL_LABEL)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake,labels).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            list_errG.append(errG.item())

        # save model
        torch.save(netG.state_dict(), Config.DIR_IMAGES_OUTPUT + '{}_G.pth'.format(epoch))
        torch.save(netD.state_dict(), Config.DIR_IMAGES_OUTPUT + '{}_D.pth'.format(epoch))

        # Check how the generator is doing by saving G's output on fixed_noise
        #show_generate_imgs(netG, fixed_noise)

        # evaluate and save generated images
        with torch.no_grad():
            dir_output = Config.DIR_IMAGES_OUTPUT + str(epoch)
            clean_dir(dir_output)
            validate_images_gen(netG, fixed_noise, fixed_labels, dir_output)
            fdi = evaluate_dataset(dir_output, mifid)

        # print
        printBoth(Config.LOG, 'epoch={}; errD={:0.5}; errG={:0.5}; fdi={:0.5}'.\
                        format(epoch, np.average(list_errD), np.average(list_errG), fdi))


if __name__ == '__main__':
    mifid = MIFID(model_path='./evaluation_script/client/motorbike_classification_inception_net_128_v4_e36.pb',
                  public_feature_path='./evaluation_script/client/public_feature.npz')

    print_params()
    train_loader = create_dataloader()
    netG, netD = create_models()
    train(train_loader, netG, netD, mifid)
