###PATH

LABELS_PATH = r'D:\Foodvisor\dataset\label'
IMGS_PATH = r'D:\Foodvisor\dataset\imgs\assignment_imgs'
CKPT_PATH = r'D:\Foodvisor\ckpt\model\model'
LOG_PATH = r'D:\Foodvisor\logs\model'

###TRAINING
TOMATO_STR = ['Tomatoes', "Tomato", "Tomatoe", "Cherry tomatoes"]
TRAIN_RATIO = 0.80
EPOCHS = 40
BATCH_SIZE = 16
LR = 10e-4
IMG_SIZE = 600

#OVERSAMPLE is used to cope with the unbalanced dataset but does not necessarily produce a better error rate on an eval set.
#However, it tends to predict more false positive (image without tomatoes labeled as True) and less false negative which
#is a better outcome when trying to warn about potentially dangerous food

OVERSAMPLE = True

EVAL_IMGS = ['1dc879c960bbf936557a4162bd39df41.jpeg',
            '247799_2018_04_15_11_34_37_170813.jpg',
            'f925994f8ac49a172a48267019bfdd58.jpeg',
            '1cb1d548bd39d3d58b5f94059d5e3ec7.jpeg',
            'b9cab18e031f11a8b0c76e6f8dd64965.jpeg',
            '08f884dcb02bfad6f6b471dd062dc1b0.jpeg',
            '948e03caf9977b750afc9fab14f89836.jpeg',
            '3510c16a6c61b3fa279d69dd61461d5d.jpeg',
            '283336_2018_04_13_10_36_11_768287.jpg',
            '239120_2018_04_21_10_12_41_697687.jpg',
            '48c2d653d121965397dafcfcdf605e33.jpeg',
            '429cf75794f58ba1c70dd989b6da0697.jpeg',
            '281281_2018_04_16_09_42_33_752693.jpg',
            'ac53d71cce0fbea0f22f7e04d15eaa65.jpeg',
            '291418_2018_04_19_09_59_14_360113.jpg',
            '8eddde5ab3f1625b3cbe9bc058d7eef9.jpeg',
            '899763bf99967d0024dbb9f097312f4e.jpeg',
            '6a2ef2c04ef66fdd30a221a762f7817f.jpeg',
            '243889_2018_04_18_19_13_10_917472.jpg',
            '4c63fdfc030b66ac39cf045e65308c18.jpeg',
            'fdf6836ec5a77146139918afea935f73.jpeg',
            '221577_2018_04_17_10_59_38_930032.jpg',
            '262793_2018_04_22_17_51_27_336586.jpg',
            'e1a81e30312b4a1f687ffe84e34f7e27.jpeg',
            '3fdb2d4bc8d14745ec87b1eb12361b41.jpeg',
            '263905_2018_04_16_06_07_59_033374.jpg']
