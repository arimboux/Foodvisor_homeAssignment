from config import *
import os
from utils import contains_tomato, split_dataset, decode_img, augment_img
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras import Model, backend
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.utils import class_weight


class TomatoDectector(Model):
    def __init__(self):
        super(TomatoDectector, self).__init__()
        self.conv1 = Conv2D(16, (3,3), activation='relu')
        self.pool1 = MaxPooling2D(pool_size = (2, 2))
        self.conv2 = Conv2D(32, (3, 3), activation='relu')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.conv3 = Conv2D(32, (3, 3), activation='relu')
        self.pool3 = MaxPooling2D(pool_size=(2, 2))
        self.conv4 = Conv2D(64, (3, 3), activation='relu')
        self.pool4 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.d1 = Dense(64, activation='relu')
        self.d2 = Dense(32, activation='relu')
        self.d3 = Dense(16, activation='relu')
        self.d4 = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = Dropout(0.1)(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = Dropout(0.1)(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = Dropout(0.3)(x)
        x = self.d2(x)
        x = Dropout(0.3)(x)
        x = self.d3(x)
        x = Dropout(0.3)(x)
        return self.d4(x)

class CustomTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

def train():

    bin_dict = contains_tomato(os.path.join(LABELS_PATH, 'img_labels.json'))

    xtrain, ytrain, xval, yval = split_dataset(bin_dict, IMGS_PATH)

    # Info about train and test set
    print(f'Train set contains {len(ytrain)} img and {ytrain.count(True)} contain tomatoes')
    print(f'Val set contains {len(yval)} img and {yval.count(True)} contain tomatoes')

    # Class weights because of unbalanced data (relevant when OVERSAMPLE is False)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(ytrain + yval), ytrain + yval)
    print(class_weights)

    file_ds_train = tf.data.Dataset.from_tensor_slices((xtrain, ytrain)).repeat()
    file_ds_val = tf.data.Dataset.from_tensor_slices((xval, yval))

    #Image decoding & augmentation
    dataset_train = file_ds_train.map(decode_img)
    dataset_val = file_ds_val.map(decode_img)
    dataset_train = dataset_train.map(augment_img)

    #Batching
    dataset_train = dataset_train.batch(BATCH_SIZE)
    dataset_val = dataset_val.batch(BATCH_SIZE)

    #Callbacks
    tensorboard_cbk = CustomTensorBoard(log_dir=LOG_PATH)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CKPT_PATH, verbose=1, save_weights_only=True, period=10)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    model = TomatoDectector()
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['BinaryAccuracy'])
    history = model.fit(dataset_train, batch_size=None, epochs=EPOCHS, verbose=2, shuffle=True,
                        steps_per_epoch=len(ytrain) // BATCH_SIZE,
                        validation_data=dataset_val,
                        callbacks=[tensorboard_cbk, cp_callback],
                        class_weight={0:class_weights[0], 1:class_weights[1]})

    # Plot training & validation accuracy values
    plt.plot(history.history['BinaryAccuracy'])
    plt.plot(history.history['val_BinaryAccuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(CKPT_PATH, '../accuracy.png'))
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(CKPT_PATH, '../loss.png'))
    plt.show()

def has_tomatoes(img_test, ckpt_path):

    model = TomatoDectector()
    model.load_weights(ckpt_path).expect_partial()
    img, _ = decode_img(img_test, True)

    img_exp = np.expand_dims(img, axis=0)
    pred = model.predict(img_exp)
    pred = float(np.squeeze(pred))
    result = round(pred)
    print(f'Img {img_test} contains tomato : {bool(result)} ({pred})')

    plt.figure()
    plt.title(f'{bool(result)}  ({str(pred)})')
    plt.imshow(img)
    plt.show()

    return bool(result)

def evaluate(ckpt_path):

    bin_dict = contains_tomato(os.path.join(LABELS_PATH, 'img_labels.json'))
    model = TomatoDectector()
    model.load_weights(ckpt_path).expect_partial()

    error_count = 0
    for img_test in EVAL_IMGS:
        img_path = os.path.join(IMGS_PATH, img_test)
        img, _ = decode_img(img_path, True)

        img_exp = np.expand_dims(img, axis=0)
        pred = model.predict(img_exp)
        pred = float(np.squeeze(pred))
        result = round(pred)
        print(f'Img {img_test} contains tomato : {bool(result)} ({pred}) gt : {bool(bin_dict[img_test])}')

        if bin_dict[img_test] != bool(result):
            error_count += 1
    print('error_rate :', error_count/len(EVAL_IMGS))