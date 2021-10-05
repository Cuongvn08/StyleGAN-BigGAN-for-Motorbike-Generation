import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2
import scipy

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
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import spectral_norm

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
MODEL_NAME = 'ralsgan_v2_2'
LOG = 'log_{}.txt'.format(MODEL_NAME)

LIMIT_DATA = -1

EPOCHS = 500
NUM_ITERATIONS = 50000
DECAY_START_ITERATION = 50000
D_STEPS = 1

BATCH_SIZE  = 32
NUM_WORKERS = 4

NC = 3
NZ = 120
NGF = 36
NDF = 40
EMBED_DIM = 32

USE_ATTN = False #True

NUM_CLASSES = 1

LR_G = 2e-4
LR_D = 4e-4

BETA1 = 0.0
BETA2 = 0.999

MARGIN = 1.0
GAMMA = 0.1
EMA = 0.999

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
            label = torch.as_tensor(label, dtype=torch.long)
            self.labels.append(label)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label
        #return {'img':img, 'label':label}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Attention(nn.Module):
    def __init__(self, channels, reduction_attn=8, reduction_sc=2):
        super().__init__()
        self.channles_attn = channels // reduction_attn
        self.channels_sc = channels // reduction_sc

        self.conv_query = spectral_norm(nn.Conv2d(channels, self.channles_attn, kernel_size=1, bias=False))
        self.conv_key = spectral_norm(nn.Conv2d(channels, self.channles_attn, kernel_size=1, bias=False))
        self.conv_value = spectral_norm(nn.Conv2d(channels, self.channels_sc, kernel_size=1, bias=False))
        self.conv_attn = spectral_norm(nn.Conv2d(self.channels_sc, channels, kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

        nn.init.orthogonal_(self.conv_query.weight.data)
        nn.init.orthogonal_(self.conv_key.weight.data)
        nn.init.orthogonal_(self.conv_value.weight.data)
        nn.init.orthogonal_(self.conv_attn.weight.data)

    def forward(self, x):
        batch, _, h, w = x.size()

        proj_query = self.conv_query(x).view(batch, self.channles_attn, -1)
        proj_key = F.max_pool2d(self.conv_key(x), 2).view(batch, self.channles_attn, -1)

        attn = torch.bmm(proj_key.permute(0,2,1), proj_query)
        attn = F.softmax(attn, dim=1)

        proj_value = F.max_pool2d(self.conv_value(x), 2).view(batch, self.channels_sc, -1)
        attn = torch.bmm(proj_value, attn)
        attn = attn.view(batch, self.channels_sc, h, w)
        attn = self.conv_attn(attn)

        out = self.gamma * attn + x

        return out

class CBN2d(nn.Module):
    def __init__(self, num_features, num_conditions):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = spectral_norm(nn.Conv2d(num_conditions, num_features*2, kernel_size=1, bias=False))

        nn.init.orthogonal_(self.embed.weight.data)

    def forward(self, x, y):
        out = self.bn(x)
        embed = self.embed(y.unsqueeze(2).unsqueeze(3))
        gamma, beta = embed.chunk(2, dim=1)
        out = (1.0 + gamma) * out + beta

        return out

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conditions, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.cbn1 = CBN2d(in_channels, num_conditions)
        self.cbn2 = CBN2d(out_channels, num_conditions)
        if self.learnable_sc:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.relu = nn.LeakyReLU()

        nn.init.orthogonal_(self.conv1.weight.data)
        nn.init.orthogonal_(self.conv2.weight.data)
        if self.learnable_sc:
            nn.init.orthogonal_(self.conv_sc.weight.data)

    def _upsample_conv(self, x, conv):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = conv(x)

        return x

    def _residual(self, x, y):
        x = self.relu(self.cbn1(x, y))
        x = self._upsample_conv(x, self.conv1) if self.upsample else self.conv1(x)
        x = self.relu(self.cbn2(x, y))
        x = self.conv2(x)

        return x

    def _shortcut(self, x):
        if self.learnable_sc:
            x = self._upsample_conv(x, self.conv_sc) if self.upsample else self.conv_sc(x)

        return x

    def forward(self, x, y):
        return self._shortcut(x) + self._residual(x, y)

class Generator(nn.Module):
    def __init__(self, latent_dim, ch, num_classes, embed_dim, use_attn=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.ch = ch
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_attn = use_attn
        self.num_chunk = 6
        num_latents = self.__get_num_latents()

        self.embed = nn.Embedding(num_classes, embed_dim)
        self.fc = spectral_norm(nn.Linear(num_latents[0], ch*16*4*4, bias=False))
        self.block1 = GBlock(ch*16, ch*16, num_latents[1], upsample=True)
        self.block2 = GBlock(ch*16, ch*8, num_latents[2], upsample=True)
        self.block3 = GBlock(ch*8, ch*4, num_latents[3], upsample=True)
        if use_attn:
            self.attn = Attention(ch*4)
        self.block4 = GBlock(ch*4, ch*2, num_latents[4], upsample=True)
        self.block5 = GBlock(ch*2, ch*1, num_latents[5], upsample=True)
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.LeakyReLU()
        self.conv_last = spectral_norm(nn.Conv2d(ch, 3, kernel_size=3, padding=1, bias=False))
        self.tanh = nn.Tanh()

        nn.init.orthogonal_(self.embed.weight.data)
        nn.init.orthogonal_(self.fc.weight.data)
        nn.init.orthogonal_(self.conv_last.weight.data)
        nn.init.constant_(self.bn.weight.data, 1.0)
        nn.init.constant_(self.bn.bias.data, 0.0)

        '''
        G x,y torch.Size([16, 120]) torch.Size([16])
        G xs 6
        G y torch.Size([16, 32])
        G h torch.Size([16, 16384])
        G block1 torch.Size([16, 1024, 8, 8])
        G block2 torch.Size([16, 512, 16, 16])
        G block3 torch.Size([16, 256, 32, 32])
        G block4 torch.Size([16, 128, 64, 64])
        G block5 torch.Size([16, 64, 128, 128])
        G out torch.Size([16, 3, 128, 128])
        '''

    def __get_num_latents(self):
        xs = torch.empty(self.latent_dim).chunk(self.num_chunk)
        num_latents = [x.size(0) for x in xs]
        for i in range(1, self.num_chunk):
            num_latents[i] += self.embed_dim

        return num_latents

    def forward(self, x, y):
        #print('G x,y', x.shape, y.shape)

        xs = x.chunk(self.num_chunk, dim=1)
        #print('G xs', len(xs))

        y = self.embed(y)
        #print('G y', y.shape)

        h = self.fc(xs[0])
        #print('G h', h.shape)

        h = h.view(h.size(0), self.ch*16, 4, 4)
        h = self.block1(h, torch.cat([y, xs[1]], dim=1))
        #print('G block1', h.shape)

        h = self.block2(h, torch.cat([y, xs[2]], dim=1))
        #print('G block2', h.shape)

        h = self.block3(h, torch.cat([y, xs[3]], dim=1))
        #print('G block3', h.shape)

        if self.use_attn:
            h = self.attn(h)

        h = self.block4(h, torch.cat([y, xs[4]], dim=1))
        #print('G block4', h.shape)

        h = self.block5(h, torch.cat([y, xs[5]], dim=1))
        #print('G block5', h.shape)

        h = self.relu(self.bn(h))
        out = self.tanh(self.conv_last(h))
        #print('G out', out.shape)

        return out

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, optimized=False):
        super().__init__()
        self.downsample = downsample
        self.optimized = optimized
        self.learnable_sc = in_channels != out_channels or downsample

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        if self.learnable_sc:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.relu = nn.LeakyReLU(0.2)

        nn.init.orthogonal_(self.conv1.weight.data)
        nn.init.orthogonal_(self.conv2.weight.data)
        if self.learnable_sc:
            nn.init.orthogonal_(self.conv_sc.weight.data)

    def _residual(self, x):
        if not self.optimized:
            x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)

        return x

    def _shortcut(self, x):
        if self.learnable_sc:
            if self.optimized:
                x = self.conv_sc(F.avg_pool2d(x, 2)) if self.downsample else self.conv_sc(x)
            else:
                x = F.avg_pool2d(self.conv_sc(x), 2) if self.downsample else self.conv_sc(x)

        return x

    def forward(self, x):
        return self._shortcut(x) + self._residual(x)

class Discriminator(nn.Module):
    def __init__(self, ch, num_classes, use_attn=False):
        super().__init__()
        self.ch = ch
        self.num_classes = num_classes
        self.use_attn = use_attn

        self.block1 = DBlock(NC, ch, downsample=True, optimized=True)
        if use_attn:
            self.attn = Attention(ch)
        self.block2 = DBlock(ch, ch*2, downsample=True)
        self.block3 = DBlock(ch*2, ch*4, downsample=True)
        self.block4 = DBlock(ch*4, ch*8, downsample=True)
        self.block5 = DBlock(ch*8, ch*16, downsample=True)
        self.relu = nn.LeakyReLU(0.2)
        self.fc = spectral_norm(nn.Linear(ch*16, 1, bias=False))
        self.embed = spectral_norm(nn.Embedding(num_classes, ch*16))
        self.clf = spectral_norm(nn.Linear(ch*16, num_classes, bias=False))

        nn.init.orthogonal_(self.fc.weight.data)
        nn.init.orthogonal_(self.embed.weight.data)
        nn.init.orthogonal_(self.clf.weight.data)

        '''
        D x,y torch.Size([16, 3, 128, 128]) torch.Size([16])
        D block1 torch.Size([16, 64, 64, 64])
        D block2 torch.Size([16, 128, 32, 32])
        D block3 torch.Size([16, 256, 16, 16])
        D block4 torch.Size([16, 512, 8, 8])
        D block5 torch.Size([16, 1024, 4, 4])
        D fc torch.Size([16, 1])
        D out torch.Size([16, 1])
        D ac torch.Size([16, 1])
        D ac torch.Size([16, 1])
        '''

    def forward(self, x, y):
        #print('D x,y', x.shape, y.shape)

        h = self.block1(x)
        #print('D block1', h.shape)

        if self.use_attn:
            h = self.attn(h)
        h = self.block2(h)
        #print('D block2', h.shape)

        h = self.block3(h)
        #print('D block3', h.shape)

        h = self.block4(h)
        #print('D block4', h.shape)

        h = self.block5(h)
        #print('D block5', h.shape)

        h = self.relu(h)
        h = torch.sum(h, dim=(2,3))

        out = self.fc(h)
        #print('D fc', out.shape)

        out += torch.sum(self.embed(y)*h, dim=1, keepdim=True)
        #print('D out', out.shape)

        ac = self.clf(h)
        #print('D ac', ac.shape)

        ac = F.log_softmax(ac, dim=1)
        #print('D ac', ac.shape)

        return out, ac



def generate_seed(manualSeed=None):
    if manualSeed is None:
        manualSeed = random.randint(1000, 10000)  # fix seed
    printBoth(LOG, 'RANDOM SEED: {}'.format(manualSeed))
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    #cudnn.benchmark = True

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
    netGE = Generator(NZ , NGF, NUM_CLASSES, EMBED_DIM, USE_ATTN).to(device, torch.float32)
    netGE.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # generate
    clean_dir(dir_images_output)
    for batch in range(int(num_images/batch_size)):
        latents = truncated_normal((batch_size, NZ), threshold=truncated, dtype=torch.float32, device=device)
        labels =  torch.randint(0, NUM_CLASSES, size=(batch_size,), dtype=torch.long, device=device)

        with torch.no_grad():
            gen_images = netGE(latents, labels).to('cpu').clone().detach().squeeze(0)
        gen_images = gen_images.to('cpu').clone().detach()
        gen_images = gen_images*0.5 + 0.5

        for i in range(gen_images.size(0)):
            save_image(gen_images[i, :, :, :], os.path.join(dir_images_output, '{}_{}.png'.format(batch, i)))

def calc_advloss_D(real, fake, margin=1.0):
    loss_real = torch.mean((real - fake.mean() - margin) ** 2)
    loss_fake = torch.mean((fake - real.mean() + margin) ** 2)
    loss = (loss_real + loss_fake) / 2

    return loss

def calc_advloss_G(real, fake, margin=1.0):
    loss_real = torch.mean((real - fake.mean() + margin) ** 2)
    loss_fake = torch.mean((fake - real.mean() - margin) ** 2)
    loss = (loss_real + loss_fake) / 2

    return loss

def sample_latents(batch_size, latent_dim, num_classes):
    latents = torch.randn((batch_size, latent_dim), dtype=torch.float32, device=device)
    labels = torch.randint(0, num_classes, size=(batch_size,), dtype=torch.long, device=device)

    return latents, labels

def validate_images_gen(netG, fixed_latents, fixed_labels, dir_output):
    gen_images = netG(fixed_latents, fixed_labels).to('cpu').clone().detach().squeeze(0)
    gen_images = gen_images*0.5 + 0.5
    for i in range(gen_images.size(0)):
        save_image(gen_images[i, :, :, :], os.path.join(dir_output, '{}.png'.format(i)))

def evaluate_dataset(dir_dataset, mifid):
    img_paths = glob(os.path.join(dir_dataset,'*.*'))

    img_np = np.empty((len(img_paths), IMG_SIZE, IMG_SIZE, NC), dtype=np.uint8)
    for idx, path in tqdm(enumerate(img_paths)):
        img_arr = cv2.imread(path)[..., ::-1]
        img_arr = np.array(img_arr)
        img_np[idx] = img_arr

    score = mifid.compute_mifid(img_np)

    return score

def truncated_normal(size, threshold=2.0, dtype=torch.float32, device='cpu'):
    x = scipy.stats.truncnorm.rvs(-threshold, threshold, size=size)
    x = torch.from_numpy(x).to(device, dtype)

    return x

if __name__ == '__main__':
    # load the evaluation model
    printBoth(LOG, 'Loading the evaluation model ...')
    mifid = MIFID(model_path='./evaluation_script/client/motorbike_classification_inception_net_128_v4_e36.pb',
                  public_feature_path='./evaluation_script/client/public_feature.npz')

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    printBoth(LOG, 'DEVICE = {}'.format(device))

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
    def get_dataiterator(device='cpu'):
        train_set = MotobikeDataset(path=DIR_IMAGES_INPUT,
                                    img_list=img_filenames,
                                    transform1=transform1,
                                    transform2=transform2,
                              )
        train_loader = DataLoader(train_set,
                                  shuffle=True,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS)
        printBoth(LOG, 'The length of train_set = {}'.format(len(train_set)))
        printBoth(LOG, 'The length of train_loader = {}'.format(len(train_loader)))

        while True:
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                imgs += (1.0 / 128.0) * torch.rand_like(imgs)

                yield imgs, labels

    train_dataiterator = get_dataiterator(device=device)

    # model
    netG = Generator(NZ , NGF, NUM_CLASSES, EMBED_DIM, USE_ATTN).to(device, torch.float32)
    netD = Discriminator(NDF, NUM_CLASSES, USE_ATTN).to(device, torch.float32)

    if PATH_MODEL_G is not '':
        netG.load_state_dict(torch.load(PATH_MODEL_G, map_location=torch.device(device)))
    if PATH_MODEL_D is not '':
        netD.load_state_dict(torch.load(PATH_MODEL_D, map_location=torch.device(device)))

    netGE = Generator(NZ , NGF, NUM_CLASSES, EMBED_DIM, USE_ATTN).to(device, torch.float32)  # Exponential moving average of generator weights works well.
    netGE.load_state_dict(netG.state_dict())

    printBoth(LOG, 'count_parameters of netG = {}'.format(count_parameters(netG)))
    printBoth(LOG, 'count_parameters of netGE = {}'.format(count_parameters(netGE)))
    printBoth(LOG, 'count_parameters of netD = {}'.format(count_parameters(netD)))

    optim_G = optim.Adam(params=netG.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optim_D = optim.Adam(params=netD.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    decay_iter = NUM_ITERATIONS - DECAY_START_ITERATION
    if decay_iter > 0:
        lr_lambda_G = lambda x: (max(0,1-x/decay_iter))
        lr_lambda_D = lambda x: (max(0,1-x/(decay_iter*D_STEPS)))
        lr_sche_G = LambdaLR(optim_G, lr_lambda=lr_lambda_G)
        lr_sche_D = LambdaLR(optim_D, lr_lambda=lr_lambda_D)

    criterion = nn.NLLLoss().to(device, torch.float32)

    optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(BETA1, BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(BETA1, BETA2))

    # train
    clean_dir(DIR_IMAGES_OUTPUT)

    fixed_latents = truncated_normal((128, NZ), dtype=torch.float32, device=device)
    fixed_labels =  torch.randint(0, NUM_CLASSES, size=(128,), dtype=torch.long, device=device)

    step = 1
    interval = 100
    while True:
        # Discriminator
        for i in range(D_STEPS):
            for param in netD.parameters():
                param.requires_grad_(True)

            optim_D.zero_grad()

            real_imgs, real_labels = train_dataiterator.__next__()

            batch_size = real_imgs.size(0)

            latents, fake_labels = sample_latents(batch_size, NZ, NUM_CLASSES)
            fake_imgs = netG(latents, fake_labels).detach()

            preds_real, preds_real_labels = netD(real_imgs, real_labels)
            preds_fake, _ = netD(fake_imgs, fake_labels)

            loss_D = calc_advloss_D(preds_real, preds_fake, MARGIN)
            loss_D += GAMMA * criterion(preds_real_labels, real_labels)
            loss_D.backward()
            optim_D.step()

            if (decay_iter > 0) and (step > DECAY_START_ITERATION):
                lr_sche_D.step()


        # Generator
        for param in netD.parameters():
            param.requires_grad_(False)

        optim_G.zero_grad()

        real_imgs, real_labels = train_dataiterator.__next__()
        batch_size = real_imgs.size(0)

        latents, fake_labels = sample_latents(batch_size, NZ, NUM_CLASSES)
        fake_imgs = netG(latents, fake_labels)

        preds_real, _ = netD(real_imgs, real_labels)
        preds_fake, preds_fake_labels = netD(fake_imgs, fake_labels)

        loss_G = calc_advloss_G(preds_real, preds_fake, MARGIN)
        loss_G += GAMMA * criterion(preds_fake_labels, fake_labels)
        loss_G.backward()
        optim_G.step()

        if (decay_iter > 0) and (step > DECAY_START_ITERATION):
            lr_sche_G.step()

        # Update Generator Eval
        for param_G, param_GE in zip(netG.parameters(), netGE.parameters()):
            param_GE.data.mul_(EMA).add_((1-EMA)*param_G.data)
        for buffer_G, buffer_GE in zip(netG.buffers(), netGE.buffers()):
            buffer_GE.data.mul_(EMA).add_((1-EMA)*buffer_G.data)

        # evaluate, log and save model
        if step % interval is 0:
            # evaluate
            with torch.no_grad():
                dir_output = DIR_IMAGES_OUTPUT + str(step)
                clean_dir(dir_output)

                validate_images_gen(netGE, fixed_latents, fixed_labels, dir_output)
                fdi = evaluate_dataset(dir_output, mifid)

            # log
            printBoth(LOG, 'step={}; loss_D={:0.5}; loss_G={:0.5}; fdi={:0.5}'.format(step, loss_D.item(), loss_G.item(), fdi))

            # save model
            torch.save(netGE.state_dict(), DIR_IMAGES_OUTPUT + '{}_G.pth'.format(step))
            torch.save(netD.state_dict(),  DIR_IMAGES_OUTPUT + '{}_D.pth'.format(step))


        # stopping
        if step < NUM_ITERATIONS:
            step += 1
        else:
            print('total step: {}'.format(step))
            break
