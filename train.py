#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import multiprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

symbols_file = open('symbols.txt', 'r')
symbols = ' ' + ''.join(sorted(symbols_file.readline().strip('\n')))
symbols_file.close()

lens = list(range(1, 7))
max_len = 8

height, width = 64, 128

decode_label = lambda s: ''.join([symbols[x] for x in s[:s.index(0)]])
encode_label = lambda s: [symbols.find(x) for x in s.ljust(max_len)]

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")

        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return loss

def create_model(len_max, n_symbols):
    input = layers.Input(shape=(width, height, 1), name="image", dtype="float32")
    label = layers.Input(name="label", shape=(len_max,), dtype="int64")

    x = input
    
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='max1')(x)

    x = layers.Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='max2')(x)

    x = layers.Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(1, 2), name='max3')(x)

    x = layers.Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', name='conv6')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(1, 2), name='max4')(x)

    x = layers.Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    model = keras.Model(input, x, name='cnn')
    x = model(input)

    conv_shape = x.get_shape()
    x = layers.Reshape((int(conv_shape[1]), int(conv_shape[3] * conv_shape[2])))(x)

    x = layers.Dense(32)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)

    rnn_size = 128

    x = layers.Bidirectional(layers.GRU(rnn_size, kernel_initializer='he_normal', return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(rnn_size, kernel_initializer='he_normal', return_sequences=True))(x)

    x = layers.Dropout(0.25)(x)
    x = layers.Dense(n_symbols + 1, kernel_initializer='he_normal', activation='softmax')(x)

    predict_model = keras.Model(input, x)

    loss_out = CTCLayer(name='ctc_loss')(label, x)

    model = keras.Model(inputs=[input, label], outputs=loss_out)
    model.compile(optimizer=keras.optimizers.Adam())

    return model, predict_model

class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_symbols = symbols
        self.captcha_width = width
        self.captcha_height = height

        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.count = len(file_list)
        self.cache = [None for _ in range(self.count)]

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        
        if self.cache[idx] != None:
            return self.cache[idx]

        X = numpy.zeros((self.batch_size, self.captcha_width, self.captcha_height, 1), dtype=numpy.float32)
        y = numpy.zeros((self.batch_size, max_len,), dtype=numpy.int64)

        for i in range(self.batch_size):
            random_image_label = random.choice(list(self.files.keys()))
            random_image_file = self.files[random_image_label]

            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            grey_data = cv2.cvtColor(numpy.array(raw_data), cv2.COLOR_BGR2GRAY)
            processed_data = grey_data / 255.0
            processed_data = numpy.expand_dims(processed_data, axis=-1)
            processed_data = processed_data.transpose(1, 0, 2)
            X[i] = processed_data

            random_image_label = random_image_label.split('_')[0]
            
            y[i] = encode_label(random_image_label)

        self.cache[idx] = {
            "image": X, 
            "label": y,
        }

        return self.cache[idx]
    
    def reset(self):
        self.cache = [None for _ in range(self.count)]
    
    def on_epoch_end(self):
        random.shuffle(self.cache)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', help='How many images in training captcha batches', type=int)
    parser.add_argument('--train-dataset', help='Where to look for the training image dataset', type=str)
    parser.add_argument('--validate-dataset', help='Where to look for the validation image dataset', type=str)
    parser.add_argument('--output-model-name', help='Where to save the trained model', type=str)
    parser.add_argument('--input-model', help='Where to look for the input model to continue training', type=str)
    parser.add_argument('--epochs', help='How many training epochs to run', type=int)
    args = parser.parse_args()

    if args.batch_size is None:
        print("Please specify the training batch size")
        exit(1)

    if args.epochs is None:
        print("Please specify the number of training epochs to run")
        exit(1)

    if args.train_dataset is None:
        print("Please specify the path to the training data set")
        exit(1)

    if args.validate_dataset is None:
        print("Please specify the path to the validation data set")
        exit(1)

    if args.output_model_name is None:
        print("Please specify a name for the trained model")
        exit(1)

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "No GPU available!"
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # with tf.device('/device:GPU:0'):
    with tf.device('/device:CPU:0'):
    # with tf.device('/device:XLA_CPU:0'):
        model, predict_model = create_model(max_len, len(symbols))

        checkpoint = args.input_model
        if checkpoint is not None:
            model.load_weights(checkpoint)

        # model.compile(loss='categorical_crossentropy',
        #               optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
        #               metrics=['accuracy'])

        model.summary()
        predict_model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(args.output_model_name+'.h5', save_best_only=False)
        ]

        training_data = ImageSequence(args.train_dataset, args.batch_size)
        validation_data = ImageSequence(args.validate_dataset, args.batch_size)

        # Save the model architecture to JSON
        with open(args.output_model_name+".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            model.fit(training_data,
                      validation_data=validation_data,
                      epochs=args.epochs,
                      callbacks=[callbacks],
                      workers=multiprocessing.cpu_count())
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + args.output_model_name+'_resume.h5')
            model.save(args.output_model_name+'_resume.h5')

if __name__ == '__main__':
    main()
