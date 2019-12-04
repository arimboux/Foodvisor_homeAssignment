import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from utils import contain_tomato
from config import *

def decode_img(img, label):

    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [600, 600])

    return img, label

model = tf.keras.models.load_model('../../ckpt')
bin_dict = contain_tomato('../../dataset/label/img_ingredients.json')

img = list(bin_dict.keys())
sample = random.sample(img, 50)
print(sample)

for img_name in EVAL_IMGS:
    path = os.path.join(IMGDIR_PATH, img_name)
    img, _ = decode_img(path, True)

    img_exp = np.expand_dims(img, axis=0)
    pred = model.predict(img_exp)
    print(pred)
    pred = float(np.squeeze(pred))
    result = round(pred)
    print(pred)
    print('Img', img_name, result, '(gt:', bool(bin_dict[img_name]), ')')

    plt.figure()
    plt.imshow(img)
    plt.show()

# test_file = r'D:\Foodvisor\dataset\imgs\assignment_imgs\1cb791b47402d1ffdfffca6c1f6cbc90.jpeg'
# img, label = decode_img(test_file, True)
# img = np.expand_dims(img, axis=0)
#
# pred = model.predict(img)
# print(pred)
