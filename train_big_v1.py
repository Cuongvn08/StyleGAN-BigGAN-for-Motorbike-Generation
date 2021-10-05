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

import sys
from evaluation_script.client.mifid_demo import MIFID
from glob import glob

import pytz
from datetime import datetime
tz = pytz.timezone('Asia/Saigon')


# set params
print('Setting params ...')
MODEL_NAME = 'big_v1'
LOG = 'log_{}.txt'.format(MODEL_NAME)

LIMIT_DATA = 100

BATCH_SIZE  = 32
NUM_WORKERS = 4
EMA = False
LABEL_NOISE = False
LABEL_NOISE_PROB = 0.1

IMG_SIZE    = 128
MEAN1,MEAN2,MEAN3 = 0.5, 0.5, 0.5
STD1,STD2,STD3    = 0.5, 0.5, 0.5

DIR_IMAGES_INPUT = '/data/cuong/data/motobike_gen/motobike/'
DIR_IMAGES_OUTPUT = '/data/cuong/result/motobike/{}/'.format(MODEL_NAME)

NUM_WORKERS = 4
INTRUDERS = [
        '2019_08_05_05_17_32_B0xS_6hHgXG_66398352_483445189138958_8195470045202604419_n_1568719912383_18787.jpg', #
        '22_honda_20Blade_20_3__1568719132927_7959.jpg', #cannot write mode CMYK as PNG
        '50_1_1547807271_1568719515097_13285.jpg',#cannot write mode CMYK as PNG
        '83_6060897e2b1d5627435b1bec2e5a9ac2_1568719487112_12907.jpg',#cannot write mode CMYK as PNG
        '94_banner_tskt_1568719223567_9195.jpg',#cannot write mode CMYK as PNG
        'Motorel38d6l1smallMotor.jpg', # truncated
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
    def __init__(self, in_channel):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channel, affine=False) #no learning parameters

    def forward(self, inputs):
        out = self.bn(inputs)
        #embed = self.embed(label.float())
        #gamma, beta = embed.chunk(2, dim=1)
        #gamma = gamma.unsqueeze(2).unsqueeze(3)
        #beta = beta.unsqueeze(2).unsqueeze(3)
        #out = gamma * out + beta
        return out

#BigGAN + leaky_relu
class ResBlock_G(nn.Module):
    def __init__(self, in_channel, out_channel, upsample=True):
        super().__init__()
        self.cbn1 = ConditionalNorm(in_channel)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='bilinear'))
        self.conv3x3_1 = nn.utils.spectral_norm(conv3x3(in_channel, out_channel)).apply(init_weight)
        self.cbn2 = ConditionalNorm(out_channel)
        self.conv3x3_2 = nn.utils.spectral_norm(conv3x3(out_channel, out_channel)).apply(init_weight)
        self.conv1x1   = nn.utils.spectral_norm(conv1x1(in_channel, out_channel)).apply(init_weight)

    def forward(self, inputs):
        # page 17, big-gan
        x  = F.relu(self.cbn1(inputs))
        x  = self.upsample(x)
        x  = self.conv3x3_1(x)
        x  = self.conv3x3_2(F.relu(self.cbn2(x)))
        x += self.conv1x1(self.upsample(inputs)) #shortcut
        return x

class Generator(nn.Module):
    def __init__(self, nz, n_feat):
        super().__init__()
        self.fc   = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(nz, 16*n_feat*4*4)).apply(init_weight)
        )
        self.res1 = ResBlock_G(16*n_feat, 16*n_feat, upsample=True)
        self.res2 = ResBlock_G(16*n_feat,  8*n_feat, upsample=True)
        #self.attn2 = Attention(8*n_feat)
        self.res3 = ResBlock_G(8*n_feat,  4*n_feat, upsample=True)
        #self.attn = Attention(4*n_feat)
        self.res4 = ResBlock_G(4*n_feat,  2*n_feat, upsample=True)
        self.res5 = ResBlock_G(2*n_feat,  1*n_feat, upsample=True)
        self.conv = nn.Sequential(
            #nn.BatchNorm2d(1*n_feat).apply(init_weight),
            nn.ReLU(),
            nn.utils.spectral_norm(conv3x3(1*n_feat, 3)).apply(init_weight),
        )

    def forward(self, z):
        batch_size = z.size(0)
        z = z.squeeze()

        x = self.fc(z) #->(*,16ch*4*4)
        x = x.view(batch_size,-1,4,4) #->(*,16ch,4,4)
        x = self.res1(x) #->(*,16ch,8,8)
        x = self.res2(x) #->(*,8ch,16,16)
        #x = self.attn2(x) #not change shape
        x = self.res3(x) #->(*,4ch,32,32)
        #x = self.attn(x) #not change shape
        x = self.res4(x) #->(*,2ch,64,64)
        x = self.res5(x) #->(*,ch,128,128)
        x = self.conv(x) #->(*,3,128,128)
        x = torch.tanh(x)
        return x


class ResBlock_D(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.utils.spectral_norm(conv3x3(in_channel, out_channel)).apply(init_weight),
            nn.ReLU(),
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
        self.res1 = ResBlock_D(3, n_feat, downsample=True)
        self.attn = Attention(n_feat)
        self.res2 = ResBlock_D(n_feat, 2*n_feat, downsample=True)
        #self.attn2 = Attention(2*n_feat)
        self.res3 = ResBlock_D(2*n_feat, 4*n_feat, downsample=True)
        self.res4 = ResBlock_D(4*n_feat, 8*n_feat, downsample=True)
        self.res5 = ResBlock_D(8*n_feat,16*n_feat, downsample=False)
        self.fc   = nn.utils.spectral_norm(nn.Linear(16*n_feat,1)).apply(init_weight)

    def forward(self, inputs):
        batch = inputs.size(0) #(*,3,128,128)
        h = self.res1(inputs) #->(*,ch,64,64)
        #h = self.attn(h) #not change shape
        h = self.res2(h) #->(*,2ch,32,32)
        #h = self.attn2(h) #not change shape
        h = self.res3(h) #->(*,4ch,16,16)
        h = self.res4(h) #->(*,8ch,8,8)
        h = self.res5(h) #->(*,16ch,8,8)
        h = torch.sum((F.relu(h)).view(batch,-1,8*8), dim=2) #GlobalSumPool ->(*,16ch)
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

    netG = Generator(nz=nz, n_feat=32).to(device) #z.shape=(*,120)
    netD = Discriminator(n_feat=36).to(device)

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
            #for _ in range(n_ite_D):
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


if __name__ == '__main__':
    # set seeds
    set_seeds()

    # create transform
    printBoth(LOG, 'Creating dataloaders ...')
    transform1 = transforms.Compose([transforms.Resize(IMG_SIZE)])

    transform2 = transforms.Compose([transforms.RandomCrop(IMG_SIZE),
                                     #transforms.RandomAffine(degrees=5),
                                     #transforms.RandomHorizontalFlip(p=0.5),
                                     #transforms.RandomApply(random_transforms, p=0.3),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[MEAN1, MEAN2, MEAN3],
                                                          std=[STD1, STD2, STD3]),
                                    ])

    img_filenames = []
    for image_name in sorted(os.listdir(DIR_IMAGES_INPUT)):
        if image_name not in INTRUDERS:
            img_filenames.append(image_name)

        if (LIMIT_DATA>0) and (len(img_filenames)>LIMIT_DATA):
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
    printBoth(LOG, 'The length of train_set = {}'.format(len(train_loader)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    printBoth(LOG, 'device={}'.format(device))

    # load pretrained model
    mifid = MIFID(model_path='./evaluation_script/client/motorbike_classification_inception_net_128_v4_e36.pb',
                  public_feature_path='./evaluation_script/client/public_feature.npz')

    # train
    res = run(lr_G=3e-4,
              lr_D=3e-4,
              beta1=0.0,
              beta2=0.999,
              nz=120,
              epochs=500,
              n_ite_D=1,
              ema_decay_rate=None,
              show_epoch_list=None,
              output_freq=10)
