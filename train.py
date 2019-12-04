from config import *
import os
from utils import contains_tomato, split_dataset, decode_img, augment_img
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras import Model


class TomatoDectector(Model):
    def __init__(self):
        super(TomatoDectector, self).__init__()
        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.pool1 = MaxPooling2D(pool_size = (2, 2))
        self.conv2 = Conv2D(32, (3, 3), activation='relu')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.d1 = Dense(32, activation='relu')
        self.d2 = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = Dropout(0.3)(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = Dropout(0.3)(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = Dropout(0.5)(x)
        return self.d2(x)

def train(imgs_path, label_path, ckpt_path):

    bin_dict = contains_tomato(label_path)

    train_set, val_set = split_dataset(bin_dict, imgs_path)
    xtrain, ytrain = zip(*train_set)
    xval, yval = zip(*val_set)


    # Info about train and test set
    print('Train set contains', len(ytrain), 'img and', ytrain.count(True), 'contain tomatoes')
    print('Val set contains', len(yval), 'img and', yval.count(True), 'contain tomatoes')


    file_ds_train = tf.data.Dataset.from_tensor_slices((list(xtrain), list(ytrain))).repeat()
    file_ds_val = tf.data.Dataset.from_tensor_slices((list(xval), list(yval)))

    dataset_train = file_ds_train.map(decode_img)
    dataset_val = file_ds_val.map(decode_img)

    dataset_train = dataset_train.map(augment_img)

    # for img, label in dataset_train.take(25):
    #     print(img.numpy())
    #     plt.figure()
    #     plt.imshow(img.numpy())
    #     plt.show()

    dataset_train = dataset_train.batch(BATCH_SIZE)
    dataset_val = dataset_val.batch(BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH)
    reduce_lr_cbk = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=10e-6)

    model = TomatoDectector()
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['BinaryAccuracy'])
    history = model.fit(dataset_train, batch_size=None, epochs=EPOCHS, verbose=1, shuffle=True,steps_per_epoch=len(ytrain) // BATCH_SIZE,
                        validation_data=dataset_val, callbacks = [tensorboard_cbk, reduce_lr_cbk])

    #Save model checkpoint
    print("Saving the model")
    model.save(ckpt_path, save_format='tf')

    # Plot training & validation accuracy values
    plt.plot(history.history['BinaryAccuracy'])
    plt.plot(history.history['val_BinaryAccuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(ckpt_path, 'accuracy.png'))
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(ckpt_path, 'loss.png'))
    plt.show()

def has_tomatoes(img_test, ckpt_path):

   model = tf.keras.models.load_model(ckpt_path)
   img, _ = decode_img(img_test, True)

   img_exp = np.expand_dims(img, axis=0)
   pred = model.predict(img_exp)
   pred = float(np.squeeze(pred))
   result = round(pred)
   print('Img', img_test, 'contains tomato :', bool(result), '(', pred, ')')

   plt.figure()
   plt.title(bool(result))
   plt.imshow(img)
   plt.show()
