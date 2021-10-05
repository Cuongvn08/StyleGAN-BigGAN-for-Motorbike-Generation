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

import pytz
from datetime import datetime
tz = pytz.timezone('Asia/Saigon')


# set params
MODEL_NAME = 'big_v4'
LOG = 'log_{}.txt'.format(MODEL_NAME)

LIMIT_DATA = -1

EPOCHS = 1000
BATCH_SIZE  = 32
NUM_WORKERS = 4

NC = 3
NZ = 150
NGF = 36
NDF = 40

LR_G=0.0001
LR_D=0.0004

BETA1 = 0.0
BETA2 = 0.999

IMG_SIZE = 128
MEAN1,MEAN2,MEAN3 = 0.5, 0.5, 0.5
STD1,STD2,STD3 = 0.5, 0.5, 0.5

DIR_IMAGES_INPUT = '/data/cuong/data/motobike_gen/motobike/'
DIR_IMAGES_OUTPUT = '/data/cuong/result/motobike/{}/'.format(MODEL_NAME)

NUM_WORKERS = 8

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

def conv3x3(in_channel, out_channel): #not change resolusion
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=3,stride=1,padding=1,dilation=1,bias=False)

def conv1x1(in_channel, out_channel): #not change resolution
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=1,stride=1,padding=0,dilation=1,bias=False)

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()

    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)

class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta    = nn.utils.spectral_norm(conv1x1(channels, channels//8)).apply(init_weight)
        self.phi      = nn.utils.spectral_norm(conv1x1(channels, channels//8)).apply(init_weight)
        self.g        = nn.utils.spectral_norm(conv1x1(channels, channels//2)).apply(init_weight)
        self.o        = nn.utils.spectral_norm(conv1x1(channels//2, channels)).apply(init_weight)
        self.gamma    = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, inputs):
        batch,c,h,w = inputs.size()
        theta = self.theta(inputs) #->(*,c/8,h,w)
        phi   = F.max_pool2d(self.phi(inputs), [2,2]) #->(*,c/8,h/2,w/2)
        g     = F.max_pool2d(self.g(inputs), [2,2]) #->(*,c/2,h/2,w/2)

        theta = theta.view(batch, self.channels//8, -1) #->(*,c/8,h*w)
        phi   = phi.view(batch, self.channels//8, -1) #->(*,c/8,h*w/4)
        g     = g.view(batch, self.channels//2, -1) #->(*,c/2,h*w/4)

        beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), -1) #->(*,h*w,h*w/4)
        o    = self.o(torch.bmm(g, beta.transpose(1,2)).view(batch,self.channels//2,h,w)) #->(*,c,h,w)
        return self.gamma*o + inputs


class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channel, affine=False) #no learning parameters
        self.embed = nn.Linear(n_condition, in_channel* 2)

        nn.init.orthogonal_(self.embed.weight.data[:, :in_channel], gain=1)
        self.embed.weight.data[:, in_channel:].zero_()

    def forward(self, inputs, label):
        out = self.bn(inputs)
        embed = self.embed(label.float())
        gamma, beta = embed.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta
        return out

#BigGAN + leaky_relu
class ResBlock_G(nn.Module):
    def __init__(self, in_channel, out_channel, condition_dim, upsample=True):
        super().__init__()
        self.cbn1 = ConditionalNorm(in_channel, condition_dim)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv3x3_1 = nn.utils.spectral_norm(conv3x3(in_channel, out_channel)).apply(init_weight)
        self.cbn2 = ConditionalNorm(out_channel, condition_dim)
        self.conv3x3_2 = nn.utils.spectral_norm(conv3x3(out_channel, out_channel)).apply(init_weight)
        self.conv1x1   = nn.utils.spectral_norm(conv1x1(in_channel, out_channel)).apply(init_weight)

    def forward(self, inputs, condition):
        x  = F.leaky_relu(self.cbn1(inputs, condition))
        x  = self.upsample(x)
        x  = self.conv3x3_1(x)
        x  = self.conv3x3_2(F.leaky_relu(self.cbn2(x, condition)))
        x += self.conv1x1(self.upsample(inputs)) #shortcut
        return x

class Generator(nn.Module):
    def __init__(self, n_feat, codes_dim=20):
        super().__init__()
        self.codes_dim = codes_dim # must be z_dim/6
        self.fc   = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(codes_dim, 16*n_feat*4*4)).apply(init_weight)
        )
        self.res1 = ResBlock_G(16*n_feat, 16*n_feat, codes_dim, upsample=True)
        self.res2 = ResBlock_G(16*n_feat,  8*n_feat, codes_dim, upsample=True)
        #self.attn2 = Attention(8*n_feat)
        self.res3 = ResBlock_G( 8*n_feat,  4*n_feat, codes_dim, upsample=True)
        self.attn = Attention(4*n_feat)
        self.res4 = ResBlock_G( 4*n_feat,  2*n_feat, codes_dim, upsample=True)
        self.res5 = ResBlock_G( 2*n_feat,  1*n_feat, codes_dim, upsample=True)
        self.conv = nn.Sequential(
            #nn.BatchNorm2d(2*n_feat).apply(init_weight),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(conv3x3(1*n_feat, NC)).apply(init_weight),
        )

    def forward(self, z):
        '''
        z.shape = (*,120)
        label_ohe.shape = (*,n_classes)
        '''
        batch = z.size(0)
        z = z.squeeze()
        codes = torch.split(z, self.codes_dim, dim=1)

        x = self.fc(codes[0]) #->(*,16ch*4*4)
        x = x.view(batch,-1,4,4) #->(*,16ch,4,4)

        condition = codes[1]
        x = self.res1(x, condition)#->(*,16ch,8,8)

        condition = codes[2]
        x = self.res2(x, condition) #->(*,8ch,16,16)
        #x = self.attn2(x) #not change shape

        condition = codes[3]
        x = self.res3(x, condition) #->(*,4ch,32,32)
        #x = self.attn(x) #not change shape

        condition = codes[4]
        x = self.res4(x, condition) #->(*,2ch,64,64)

        condition = codes[5]
        x = self.res5(x, condition) #->(*,1ch,128,128)

        x = self.conv(x) #->(*,3,128,128)
        x = torch.tanh(x)
        return x


class ResBlock_D(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(conv3x3(in_channel, out_channel)).apply(init_weight),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(conv3x3(out_channel, out_channel)).apply(init_weight),
        )
        self.shortcut = nn.Sequential(
            nn.utils.spectral_norm(conv1x1(in_channel,out_channel)).apply(init_weight),
        )
        if downsample:
            self.layer.add_module('avgpool',nn.AvgPool2d(kernel_size=2,stride=2))
            self.shortcut.add_module('avgpool',nn.AvgPool2d(kernel_size=2,stride=2))

    def forward(self, inputs):
        x  = self.layer(inputs)
        x += self.shortcut(inputs)
        return x


class Discriminator(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.res1 = ResBlock_D(NC, n_feat, downsample=True)
        #self.attn = Attention(n_feat)
        self.res2 = ResBlock_D(n_feat, 2*n_feat, downsample=True)
        self.attn = Attention(2*n_feat)
        self.res3 = ResBlock_D(2*n_feat, 4*n_feat, downsample=True)
        self.res4 = ResBlock_D(4*n_feat, 8*n_feat, downsample=True)
        self.res5 = ResBlock_D(8*n_feat,16*n_feat, downsample=True)
        self.fc   = nn.utils.spectral_norm(nn.Linear(16*n_feat,1)).apply(init_weight)
        #self.embedding = nn.Embedding(num_embeddings=n_classes, embedding_dim=16*n_feat).apply(init_weight)

    def forward(self, inputs):
        batch = inputs.size(0) #(*,3,128,128)
        h = self.res1(inputs) #->(*,ch,64,64)
        h = self.res2(h) #->(*,2ch,32,32)
        #h = self.attn(h) #not change shape
        h = self.res3(h) #->(*,4ch,16,16)
        h = self.res4(h) #->(*,8ch,8,8)
        h = self.res5(h) #->(*,16ch,4,4)
        h = torch.sum((F.leaky_relu(h,0.2)).view(batch,-1,4*4), dim=2) #GlobalSumPool ->(*,16ch)
        outputs = self.fc(h) #->(*,1)

        #if label is not None:
        #    embed = self.embedding(label) #->(*,16ch)
        #    outputs += torch.sum(embed*h,dim=1,keepdim=True) #->(*,1)

        outputs = torch.sigmoid(outputs)
        return outputs


#random seeds
def set_seeds():
    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.backends.cudnn.benchmark = True
    printBoth(LOG, 'manualSeed={}'.format(manualSeed))

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

#BigGAN
def run(lr_G=3e-4,lr_D=6e-4, beta1=0.0, beta2=0.999, nz=120, epochs=2,
        n_ite_D=1, ema_decay_rate=0.999, show_epoch_list=None, output_freq=10):

    netG = Generator(n_feat=NGF, codes_dim=20).to(device) #z.shape=(*,120)
    netD = Discriminator(n_feat=NDF).to(device)

    printBoth(LOG, 'count_parameters of netG = {}'.format(count_parameters(netG)))
    printBoth(LOG, 'count_parameters of netD = {}'.format(count_parameters(netD)))

    real_label = 0.9
    fake_label = 0

    D_loss_list = []
    G_loss_list = []

    dis_criterion = nn.BCELoss().to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, beta2))

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    netG.train()
    netD.train()

    ### training here
    printBoth(LOG, 'Starting training ...')
    clean_dir(DIR_IMAGES_OUTPUT)
    for epoch in range(1,epochs+1):
        loss_d_real = 0
        loss_d_fake = 0
        loss_g = 0
        acc_d_real = 0
        acc_d_fake = 0
        for ii, data in enumerate(train_loader):
            ############################
            # (1) Update D network
            ###########################
            # train with real
            netD.zero_grad()
            real_images = data['img'].to(device, non_blocking=True)
            batch_size  = real_images.size(0)
            dis_labels  = torch.full((batch_size, 1), 0.9, device=device) #shape=(*,)
            dis_output = netD(real_images) #dis shape=(*,1)
            errD_real  = dis_criterion(dis_output, dis_labels)
            errD_real.backward()

            loss_d_real += errD_real.item() / len(train_loader)
            acc_d_real += get_accuracy(dis_output, dis_labels) / len(train_loader)

            # train with fake
            noise  = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            dis_labels.fill_(0.0)
            dis_output = netD(fake.detach())
            errD_fake  = dis_criterion(dis_output, dis_labels)
            errD_fake.backward()
            optimizerD.step()

            loss_d_fake += errD_fake.item() / len(train_loader)
            acc_d_fake += get_accuracy(dis_output, dis_labels) / len(train_loader)

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            dis_labels.fill_(0.9)  # fake labels are real for generator cost
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake  = netG(noise)
            dis_output = netD(fake)
            errG = dis_criterion(dis_output, dis_labels)
            errG.backward()
            optimizerG.step()

            loss_g += errG.item()/len(train_loader)

        # save model
        torch.save(netG.state_dict(), DIR_IMAGES_OUTPUT + '{}_G.pth'.format(epoch))
        torch.save(netD.state_dict(), DIR_IMAGES_OUTPUT + '{}_D.pth'.format(epoch))

        # evaluate and save generated images
        with torch.no_grad():
            dir_output = DIR_IMAGES_OUTPUT + str(epoch)
            clean_dir(dir_output)

            validate_images_gen(netG, fixed_noise, dir_output)
            eval_fdi = evaluate_dataset(dir_output, mifid)

        # print
        printBoth(LOG, 'epoch={}; loss_d_real={:0.5}; loss_d_fake={:0.5}; loss_g={:0.5}; acc_d_real={:0.5}; acc_d_fake={:0.5}; eval_fdi={:0.5}'.\
                                    format(epoch, loss_d_real, loss_d_fake, loss_g, acc_d_real, acc_d_fake, eval_fdi))


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

    printBoth(LOG, 'IMG_SIZE = {}'.format(IMG_SIZE))
    printBoth(LOG, 'MEAN1 = {}; MEAN2 = {}; MEAN3 = {}'.format(MEAN1, MEAN2, MEAN3))
    printBoth(LOG, 'STD1 = {}; STD2 = {}; STD3 = {};'.format(STD1, STD2, STD3))

    printBoth(LOG, 'DIR_IMAGES_INPUT = {}'.format(DIR_IMAGES_INPUT))
    printBoth(LOG, 'DIR_IMAGES_OUTPUT = {}'.format(DIR_IMAGES_OUTPUT))

    printBoth(LOG, 'NUM_WORKERS = {}'.format(NUM_WORKERS))


def generate_images(model_path, dir_images_output, num_images=10000, batch_size=1000, truncated=None, device='cuda'):
    # load model
    netG = Generator(n_feat=NGF, codes_dim=20).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    #netG = netG.to(device)

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

            z = torch.from_numpy(z[:batch_size*NZ]).view(batch_size, NZ, 1, 1)
            z = z.float().to(device)
        else:
            z = torch.randn(batch_size, NZ, 1, 1, device=device)

        gen_images = netG(z)
        gen_images = gen_images.to(device).clone().detach().squeeze(0)
        gen_images = gen_images*0.5 + 0.5

        for i in range(gen_images.size(0)):
            save_image(gen_images[i, :, :, :], os.path.join(dir_images_output, '{}_{}.png'.format(batch, i)))


if __name__ == '__main__':
    # load the evaluation model
    printBoth(LOG, 'Loading the evaluation model ...')
    mifid = MIFID(model_path='./evaluation_script/client/motorbike_classification_inception_net_128_v4_e36.pb',
                  public_feature_path='./evaluation_script/client/public_feature.npz')

    # set seeds
    generate_seed()

    # params
    print_params()

    # create transform
    printBoth(LOG, 'Creating dataloaders ...')
    transform1 = transforms.Compose([transforms.Resize(IMG_SIZE)])

    transform2 = transforms.Compose([transforms.RandomCrop(IMG_SIZE),
                                     #transforms.RandomAffine(degrees=5),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     #transforms.RandomApply(random_transforms, p=0.3),
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
    printBoth(LOG, 'device={}'.format(device))

    # train
    res = run(lr_G=LR_G,
              lr_D=LR_D,
              beta1=BETA1,
              beta2=BETA2,
              nz=NZ,
              epochs=EPOCHS)
