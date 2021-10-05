# import
import os
import shutil
import random
import zipfile
import importlib
import numpy as np


def generate_seed(manualSeed=None):
    if manualSeed is None:
        manualSeed = random.randint(1000, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)

def clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def create_zip_file(dir_images, path_zip):
    def zipdir(path, ziph):
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    zipf = zipfile.ZipFile(path_zip, 'w', zipfile.ZIP_DEFLATED)
    zipdir(dir_images, zipf)
    zipf.close()


# must set the below params according to the comments at the end
TRUNCATION  = 0.7
MANUAL_SEED = 8395
MODEL_NAME  = 'ralsgan_v2'
EPOCH       = 39900

# fixed setting
#DEVICE = 'cpu'
DEVICE = 'cuda'
NUM_IMAGES  = 10000
BATCH_SIZE  = 50

PATH_MODEL_GENERATOR = './model/saved_models/{}/{}_G.pth'.format(MODEL_NAME, EPOCH)
DIR_IMAGES_OUTPUT    = './images/'
PATH_ZIP_FILE        = './images.zip'

MODEL = mymodule = importlib.import_module('train_' + MODEL_NAME)

# clean output dir
clean_dir('./output/')

# generate images
print('Generating images ...')
MODEL.generate_images(PATH_MODEL_GENERATOR,
                      DIR_IMAGES_OUTPUT,
                      num_images=NUM_IMAGES,
                      batch_size=BATCH_SIZE,
                      truncated=TRUNCATION,
                      device=DEVICE)

# zip
print('Creating zip file ...')
create_zip_file(DIR_IMAGES_OUTPUT, PATH_ZIP_FILE)

print('The end.')




'''
1.
    File_1574645614
    At 25/11/2019 08:33:34

    2019-11-25 08:23:44 Generating images ralsgan_v4
    2019-11-25 08:23:44 Random Seed: 2266
    2019-11-25 08:25:30 truncate=0.8; epoch = 44750; FDI = 43.395

2.
    File_1574595190
    At 24/11/2019 18:33:10

    2019-11-24 18:15:37 Generating images ralsgan_v2_1
    2019-11-24 18:15:37 Random Seed: 8395
    2019-11-24 18:17:59 truncate=0.8; epoch = 750; FDI = 39.704

3.
    File_1574504286
    At 23/11/2019 17:18:06

    2019-11-23 13:13:09 Generating images ralsgan_v2
    2019-11-23 13:13:09 Random Seed: 8395
    2019-11-23 13:22:25 truncate=0.7; epoch = 39900; FDI = 37.162

4.
    File_1574471002
    At 23/11/2019 08:03:22

    2019-11-23 07:24:48 Generating images ralsgan_v2
    2019-11-23 07:24:48 Random Seed: 8395
    2019-11-23 07:50:11 truncate=1.0; epoch = 41750; FDI = 38.612

5.
    File_1574350744
    At 21/11/2019 22:39:04

    019-11-21 21:57:18 Generating images ralsgan_v1
    2019-11-21 21:57:18 Random Seed: 8895
    2019-11-21 21:59:40 truncate=0.5; epoch = 22700; FDI = 41.313

6.
    File_1574125831
    At 19/11/2019 08:10:31

    2019-11-19 08:01:24 Generating images dc_style_v1_3_1
    2019-11-19 08:01:24 Random Seed: 2547
    2019-11-19 08:03:11 truncate=1.0; epoch = 119; FDI = 65.609

7.
    File_1574125629
    At 19/11/2019 08:07:09

    2019-11-19 07:57:19 Generating images dc_style_v1_3
    2019-11-19 07:57:19 Random Seed: 7005
    2019-11-19 07:59:11 truncate=0.7; epoch = 287; FDI = 64.139

8.
    File_1573796971
    At 15/11/2019 12:49:31

    2019-11-15 12:36:43 Generating images dc_style_v1
    2019-11-15 12:36:43 Random Seed: 9701
    2019-11-15 12:38:31 truncate=0.9; epoch = 100; FDI = 83.402

9.
    File_1573518087
    At 12/11/2019 07:21:27

    2019-11-11 21:46:18 Generating images big_style_v5_1
    2019-11-11 21:46:18 Random Seed: 4850
    2019-11-11 21:55:40 truncate=0.6; epoch = 77; FDI = 67.645

10.
    File_1573367367
    At 10/11/2019 13:29:27

    2019-11-10 13:07:39 Generating images big_v12_4_1
    2019-11-10 13:07:39 Random Seed: 9005
    2019-11-10 13:12:06 truncate=1.1; epoch = 209; FDI = 47.421

11.
    File_1573293186
    At 09/11/2019 16:53:06

    2019-11-09 14:39:40 Generating images big_v12_6_1
    2019-11-09 14:39:40 Random Seed: 2179
    2019-11-09 15:17:26 truncate=1.0; epoch = 132; FDI = 51.877

12.
    File_1573265478
    At 09/11/2019 09:11:18

    2019-11-09 07:27:27 Generating images big_v12_4_10
    2019-11-09 07:27:27 Random Seed: 1588
    2019-11-09 07:41:56 truncate=1.2; epoch = 143; FDI = 53.358

13.
    File_1573231826
    At 08/11/2019 23:50:26

    2019-11-08 23:03:17 Generating images big_v12_8
    2019-11-08 23:03:17 Random Seed: 4175
    2019-11-08 23:34:04 truncate=1.6; epoch = 252; FDI = 58.07

14.
    File_1573123948
    At 07/11/2019 17:52:28

    2019-11-07 13:42:21 Generating images big_v12_6
    2019-11-07 13:42:21 Random Seed: 4398
    2019-11-07 13:47:05 truncate=1.4; epoch = 128; FDI = 55.29

15.
    File_1573114609
    At 07/11/2019 15:16:49

    2019-11-07 15:10:55 Generating images big_v12_4_1
    2019-11-07 15:10:55 Random Seed: 9005
    2019-11-07 15:13:22 truncate=1.2; epoch = 201; FDI = 45.469

16.
    File_1573027705
    At 06/11/2019 15:08:25

    2019-11-06 14:17:00 Generating images big_v12_4_7
    2019-11-06 14:17:00 Random Seed: 2439
    2019-11-06 15:00:32 truncate=1.2; epoch = 141; FDI = 62.131

17.
    File_1573001392
    At 06/11/2019 07:49:52

    2019-11-06 07:38:22 Generating images big_v12_4_5
    2019-11-06 07:38:22 Random Seed: 3169
    2019-11-06 07:42:45 truncate=1.0; epoch = 205; FDI = 48.048

18.
    File_1572924339
    At 05/11/2019 10:25:39

    2019-11-05 10:16:48 Generating images big_v12_4_2; truncated=1.2
    2019-11-05 10:16:48 Random Seed: 2653
    2019-11-05 10:18:44 epoch = 215; FDI = 59.761

19.
    File_1572919158
    At 05/11/2019 08:59:18

    2019-11-05 08:17:29 Generating images big_v12_4_1; truncated=1.1
    2019-11-05 08:17:29 Random Seed: 9005
    2019-11-05 08:21:00 epoch = 146; FDI = 44.39

20.
    File_1572882951
    At 04/11/2019 22:55:51

    2019-11-04 22:31:14 Generating images big_v12_4; truncated=1.2
    2019-11-04 22:31:14 Random Seed: 9046
    2019-11-04 22:33:51 epoch = 244; FDI = 56.824

21.
    File_1572881181
    At 04/11/2019 22:26:21

    2019-11-04 22:06:51 Generating images big_v12_4; truncated=1.2
    2019-11-04 22:06:51 Random Seed: 9046
    2019-11-04 22:08:40 epoch = 240; FDI = 53.021

22.
    File_1572767302
    At 03/11/2019 14:48:22

    2019-11-03 14:25:40 Generating images big_v12; truncated=0.9
    2019-11-03 14:25:40 Random Seed: 5624
    2019-11-03 14:28:09 epoch = 139; FDI = 68.164

23.
    File_1572752331
    At 03/11/2019 10:38:51

    2019-11-03 10:11:39 Generating images big_v11; truncated=1.0
    2019-11-03 10:11:39 Random Seed: 2570
    2019-11-03 10:13:30 epoch = 279; FDI = 92.25

24.
    File_1572689231
    At 02/11/2019 17:07:11

    2019-11-02 07:42:53 Generating images big_v6; truncated=0.8
    2019-11-02 07:44:46 epoch = 120; FDI = 102.76

25.
    File_1572627408
    At 01/11/2019 23:56:48

    2019-11-01 23:39:31 Generating images big_v4; truncated=2.0
    2019-11-01 23:41:16 epoch = 107; FDI = 108.32

26.
    File_1572569083
    At 01/11/2019 07:44:43

    TRUNCATED = 0.6 MANUAL_SEED = 2428 MODEL_NAME = 'sa_v3'
    2019-10-31 19:22:26 epoch = 71; FDI = 86.997

27.
    File_1572502233
    At 31/10/2019 13:10:33

    TRUNCATED = None
    MANUAL_SEED=2428
    MODEL_NAME = 'sa_v3'
    2019-10-29 23:17:16 epoch=71;

28.
    File_1572323142
    At 29/10/2019 11:25:42

    manualSeed = 9636
    2019-10-29 11:00:31 Generating images dc_v6; truncated=0.7

'''
