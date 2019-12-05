#data augmentation
#

import argparse
from config import *
import os
from utils import convert_json
from train import train, has_tomatoes, evaluate

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--img_test', type=str, help = 'path to test image', default=os.path.join(IMGS_PATH, 'b9cab18e031f11a8b0c76e6f8dd64965.jpeg'))
    parser.add_argument('--eval', action='store_true', help='eval the network on a set of images removed from the dataset')

    args = parser.parse_args()

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    if not os.path.exists(os.path.join(LABELS_PATH, 'img_labels.json')):
        convert_json(LABELS_PATH)

    if args.train:
        train()

    if args.test:
        has_tomatoes(os.path.join(IMGS_PATH,args.img_test), CKPT_PATH)

    if args.eval:
        evaluate(CKPT_PATH)