#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import pandas
import string
import random
import argparse
import time
import tensorflow as tf
import tensorflow.keras as keras

symbols_file = open('symbols.txt', 'r')
symbols = ' ' + ''.join(sorted(symbols_file.readline().strip('\n')))
symbols_file.close()

lens = list(range(1, 7))
max_len = 8

height, width = 64, 128

decode_label = lambda s: ''.join([symbols[x] for x in s[:s.index(0)]])
encode_label = lambda s: [symbols.find(x) for x in s.ljust(max_len)]

def decode_prediction(y_pred):
    input_length = numpy.ones(y_pred.shape[0]) * y_pred.shape[1]
    y_pred = keras.backend.ctc_decode(y_pred, input_length, greedy=False)[0][0][:, :max_len]
    y_pred = [decode_label(list(y)) for y in y_pred]
    return y_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--captcha-csv', help='Where to read the captchas csv to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)
    
    if args.captcha_csv is None:
        print("Please specify the Captcha CSV break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    print("Classifying captchas with symbol set {" + symbols + "}")

    with tf.device('/cpu:0'):
        with open(args.output, 'w') as output_file:
            json_file = open(args.model_name+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights(args.model_name+'.h5')

            df = pandas.read_csv(args.captcha_csv, header=None, index_col=False, names=['filename'])[['filename']]
            df['result'] = ''

            start_time = time.time()
            for idx, row in df.iterrows():
                filename = row['filename']
                filename = os.path.join(args.captcha_dir, filename)

                image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                image = image / 255.0
                image = numpy.expand_dims(image, axis=-1)
                image = numpy.transpose(1, 0, 2) 

                image = numpy.expand_dims(image, axis=0)
                prediction = model.predict(image)
                prediction = decode_prediction(prediction)[0]
                
                print('%s: %s' % (os.path.basename(filename), prediction))

                row['result'] = prediction

                if idx % 100 == 0:
                    print('predicted %d of %d images in %f seconds' % (idx, df.shape[0], time.time() - start_time))
                
            print('predicted %d images in %f seconds' % (df.shape[0], time.time() - start_time))
    
    df.sort_values(by=['filename'], ascending=True)
    df.to_csv(output_file, columns=['filename', 'result'], header=False, index=False)

if __name__ == '__main__':
    main()
