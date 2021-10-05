#https://www.kaggle.com/alvaroma/dcgan-lb-30-65

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


# config
class Config():
    LIMIT_DATA = -1

    MODEL_NAME = 'dc_v1'
    LOG = 'log_{}.txt'.format(MODEL_NAME)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NGPU = 1

    IMAGE_SIZE = 128
    MEAN1,MEAN2,MEAN3 = 0.5, 0.5, 0.5
    STD1,STD2,STD3    = 0.5, 0.5, 0.5

    EVAL_STEP = 1 # should be 1
    BATCH_SIZE  = 64
    NUM_WORKERS = 4
    EPOCHES = 500

    LR_G = 0.0002
    LR_D = 0.0004

    BETA1 = 0.0
    BETA2 = 0.9

    LRFACTOR_G = 0.9
    LRFACTOR_D = 0.9

    NC = 3 # Number of channels in the training images. For color images this is 3
    NZ = 100 # Size of z latent vector (i.e. size of generator input)
    NGF = 64 # Size of feature maps in generator
    NDF = 64 # Size of feature maps in discriminator

    REAL_LABEL = 0.9
    FAKE_LABEL = 0.0

    PATH_PRETRAINED_G = ''
    PATH_PRETRAINED_D = ''

    DIR_IMAGES_INPUT = '/data/cuong/data/motobike_gen/motobike/'
    PATH_IMAGES_OUTPUT = '/data/cuong/result/motobike/{}/'.format(MODEL_NAME)

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

def evaluate_dataset(dir_dataset, mifid):
    img_paths = glob(os.path.join(dir_dataset,'*.*'))

    img_np = np.empty((len(img_paths), 128, 128, 3), dtype=np.uint8)
    for idx, path in tqdm(enumerate(img_paths)):
        img_arr = cv2.imread(path)[..., ::-1]
        img_arr = np.array(img_arr)
        img_np[idx] = img_arr

    score = mifid.compute_mifid(img_np)
    #print('dir_dataset={}; FID={}'.format(dir_dataset, score))
    return score

def validate_images_gen(netG, fixed_noise, dir_output):
    gen_images = netG(fixed_noise).to('cpu').clone().detach().squeeze(0)
    gen_images = gen_images*0.5 + 0.5
    for i in range(gen_images.size(0)):
        save_image(gen_images[i, :, :, :], os.path.join(dir_output, '{}.png'.format(i)))


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

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img = self.imgs[idx]
        return {'img':img}


# model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution

            nn.ConvTranspose2d( Config.NZ, Config.NGF * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(Config.NGF * 16),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4

            nn.ConvTranspose2d( Config.NGF * 16, Config.NGF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NGF * 8),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8

            nn.ConvTranspose2d(Config.NGF * 8, Config.NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NGF * 4),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16

            nn.ConvTranspose2d( Config.NGF * 4, Config.NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NGF * 2),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32

            nn.ConvTranspose2d( Config.NGF * 2, Config.NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NGF),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.ReLU(True),
            # state size. (ngf) x 64 x 64

            nn.ConvTranspose2d( Config.NGF, Config.NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128

            nn.Conv2d(Config.NC, Config.NDF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NDF),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 64 x 64

            nn.Conv2d(Config.NDF, Config.NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NDF * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 32 x 32

            nn.Conv2d(Config.NDF * 2, Config.NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NDF * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*4) x 16 x 16

            nn.Conv2d(Config.NDF * 4, Config.NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NDF * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 8 x 8

            nn.Conv2d(Config.NDF * 8, Config.NDF * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.NDF * 16),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 8 x 8

            nn.Conv2d(Config.NDF * 16, 1, 4, 1, 0, bias=False),

            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

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

    printBoth(Config.LOG, 'EVAL_STEP={}'.format(Config.EVAL_STEP))
    printBoth(Config.LOG, 'BATCH_SIZE={}'.format(Config.BATCH_SIZE))
    printBoth(Config.LOG, 'NUM_WORKERS={}'.format(Config.NUM_WORKERS))
    printBoth(Config.LOG, 'EPOCHES={}'.format(Config.EPOCHES))

    printBoth(Config.LOG, 'LR_G={}'.format(Config.LR_G))
    printBoth(Config.LOG, 'LR_D={}'.format(Config.LR_D))

    printBoth(Config.LOG, 'BETA1={}'.format(Config.BETA1))
    printBoth(Config.LOG, 'BETA2={}'.format(Config.BETA2))

    printBoth(Config.LOG, 'LRFACTOR_G={}'.format(Config.LRFACTOR_G))
    printBoth(Config.LOG, 'LRFACTOR_D={}'.format(Config.LRFACTOR_D))

    printBoth(Config.LOG, 'NC={}'.format(Config.NC))
    printBoth(Config.LOG, 'NZ={}'.format(Config.NZ))
    printBoth(Config.LOG, 'NGF={}'.format(Config.NGF))
    printBoth(Config.LOG, 'NDF={}'.format(Config.NDF))

    printBoth(Config.LOG, 'REAL_LABEL={}'.format(Config.REAL_LABEL))
    printBoth(Config.LOG, 'FAKE_LABEL={}'.format(Config.FAKE_LABEL))

    printBoth(Config.LOG, 'PATH_PRETRAINED_G={}'.format(Config.PATH_PRETRAINED_G))
    printBoth(Config.LOG, 'PATH_PRETRAINED_D={}'.format(Config.PATH_PRETRAINED_D))

    printBoth(Config.LOG, 'DIR_IMAGES_INPUT={}'.format(Config.DIR_IMAGES_INPUT))
    printBoth(Config.LOG, 'PATH_IMAGES_OUTPUT={}'.format(Config.PATH_IMAGES_OUTPUT))


def create_dataloader():
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

    # create loader
    printBoth(Config.LOG, 'Generating dataloader ...')
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
    # Create the generator
    netG = Generator(Config.NGPU).to(Config.DEVICE)

    if Config.PATH_PRETRAINED_G is not '':
        netG.load_state_dict(torch.load(Config.PATH_PRETRAINED_G, map_location=Config.DEVICE))
        printBoth(Config.LOG, 'Completed to load pretrained generator model')
    else:
        netG.apply(weights_init) #initialize all weights to mean=0, stdev=0.2.
        printBoth(Config.LOG, 'Completed to create Generator model')

    if (Config.DEVICE.type == 'cuda') and (Config.NGPU > 1):
        netG = nn.DataParallel(netG, list(range(Config.NGPU)))

    printBoth(Config.LOG, 'count_params of netG = {}'.format(count_parameters(netG)))

    # Create the Discriminator
    netD = Discriminator(Config.NGPU).to(Config.DEVICE)

    if Config.PATH_PRETRAINED_D is not '':
        netD.load_state_dict(torch.load(Config.PATH_PRETRAINED_D, map_location=Config.DEVICE))
        printBoth(Config.LOG, 'Completed to load pretrained discriminator model')
    else:
        netD.apply(weights_init) #initialize all weights to mean=0, stdev=0.2.
        printBoth(Config.LOG, 'Completed to create Discriminator model')

    if (Config.DEVICE.type == 'cuda') and (Config.NGPU > 1):
        netD = nn.DataParallel(netD, list(range(Config.NGPU)))

    printBoth(Config.LOG, 'count_params of netD = {}'.format(count_parameters(netD)))

    return netG, netD


def train(train_loader, netG, netD, mifid):
    printBoth(Config.LOG, 'Start training ...')

    # train
    criterion = nn.BCELoss() # Initialize BCELoss function
    fixed_noise = torch.randn(64, Config.NZ, 1, 1, device=Config.DEVICE) # Create batch of latent vectors that we will use to visualize

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=Config.LR_D, betas=(Config.BETA1, Config.BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=Config.LR_G, betas=(Config.BETA1, Config.BETA2))

    schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lambda epoch: Config.LRFACTOR_G ** epoch)
    schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lambda epoch: Config.LRFACTOR_D ** epoch)

    # Lists to keep track of progress
    clean_dir(Config.PATH_IMAGES_OUTPUT)

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
            real_cpu = data['img'].to(Config.DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), Config.REAL_LABEL, device=Config.DEVICE)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, Config.NZ, 1, 1, device=Config.DEVICE)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(Config.FAKE_LABEL)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
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
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            list_errG.append(errG.item())

        # adjust learning rate
        schedulerG.step()
        schedulerD.step()

        # save model
        torch.save(netG.state_dict(), Config.PATH_IMAGES_OUTPUT + '{}_G.pth'.format(epoch))
        torch.save(netD.state_dict(), Config.PATH_IMAGES_OUTPUT + '{}_D.pth'.format(epoch))

        # Check how the generator is doing by saving G's output on fixed_noise
        #show_generate_imgs(netG, fixed_noise)

        # evaludate and save generated images
        with torch.no_grad():
            dir_output = Config.PATH_IMAGES_OUTPUT + str(epoch)
            clean_dir(dir_output)
            validate_images_gen(netG, fixed_noise, dir_output)
            fdi = evaluate_dataset(dir_output, mifid)

        # print
        date_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S ')
        printBoth(Config.LOG, 'epoch={}; errD={:0.5}; errG={:0.5}; fdi={:0.5}'.\
                        format(epoch, np.average(list_errD), np.average(list_errG), fdi))


if __name__ == '__main__':
    mifid = MIFID(model_path='./evaluation_script/client/motorbike_classification_inception_net_128_v4_e36.pb',
                  public_feature_path='./evaluation_script/client/public_feature.npz')

    print_params()
    train_loader = create_dataloader()
    netG, netD = create_models()
    train(train_loader, netG, netD, mifid)
