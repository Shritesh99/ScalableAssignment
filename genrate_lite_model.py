#!/usr/bin/env python3

import os
import numpy
import random
import string
import random
import argparse
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

def convert_to_tflite(model, quantization='dr'):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    json_file = open(args.model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(args.model_name+'.h5')
    model.compile(optimizer=keras.optimizers.Adam())

    lite_model = convert_to_tflite(model)

    with open(args.output, 'wb') as f:
        f.write(lite_model)


