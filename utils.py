from config import *
import os
import json
import csv
import tensorflow as tf

def search_in_csvFile(id, path):

    with open(os.path.join(path, 'label_mapping.csv')) as labels_csv:
        labels = csv.reader(labels_csv, delimiter=',')

        for row in labels:
            if id == row[0]:
                return row[2]

def convert_json(json_path):
    """Convert json with all information into a new one containing only name of classes in an image"""

    with open(os.path.join(json_path, 'img_annotations.json')) as data:
        data = json.load(data)

    classes_dict = {}
    for img in data:
        class_list = []
        for ingredient in data[img]:
            class_list.append(search_in_csvFile(ingredient['id'], json_path))
        classes_dict[img] = class_list

    with open(os.path.join(json_path, 'img_labels.json'), 'w') as new_json:
        json.dump(classes_dict, new_json)

def contains_tomato(simplified_json):
    """Using a simplified json, produce a dict that inform if an image contains tomato"""

    with open(simplified_json) as data:
        data = json.load(data)

    for img in data:
        if any(i in data[img] for i in TOMATO_STR):
            data[img] = 1
        else:
            data[img] = 0
    return data

def decode_img(img, label):

    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [256, 256])

    return img, label

def augment_img(img, label):

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    rand = tf.random.uniform([])
    if rand > 0.5:
        img = tf.image.rot90(img, k=tf.random.uniform([], minval=1, maxval=4, dtype=tf.dtypes.int32))

    return img, label

def split_dataset(binary_dict, imgs_path):
    """Remove the images used for the evaluation part and split the dataset into a train and validation set"""

    #Remove evaluation images
    for eval_img in EVAL_IMGS:
        del binary_dict[eval_img]

    total_length = len(binary_dict)
    th = int(TRAIN_RATIO * total_length)
    full_set = []
    #Create list of (img, bool) tuple
    for img in binary_dict:
        full_set.append((os.path.join(imgs_path, img), binary_dict[img]))

    train_set = full_set[:th]
    val_set = full_set[th:]

    return train_set, val_set