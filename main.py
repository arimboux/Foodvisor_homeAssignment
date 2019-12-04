#data augmentation
#

import argparse
from config import *
import os
from utils import convert_json
from train import train, has_tomatoes

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, help='path to label files', default=LABELS_PATH)
    parser.add_argument('-p', '--image_path', type=str, help='path to dataset', default=IMGS_PATH)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('-ckpt', '--checkpoint', type=str, help='path to checkpoint', default=SAVEDIR_PATH)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--img_test', type=str, help = 'path to test image', default=os.path.join(IMGS_PATH, 'b9cab18e031f11a8b0c76e6f8dd64965.jpeg'))

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.json_path, 'img_labels.json')):
        convert_json(args.json_path)

    if args.train:
        train(args.image_path, os.path.join(args.json_path, 'img_labels.json'), args.checkpoint)

    if args.test:
        has_tomatoes(os.path.join(args.image_path,args.img_test), args.checkpoint)
