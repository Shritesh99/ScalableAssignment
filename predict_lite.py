#!/usr/bin/env python3

import cv2
import numpy as np
import os
import pandas as pd
import string
import argparse
from tflite_runtime.interpreter import Interpreter
import time


symbols_file = open('symbols.txt', 'r')
symbols = ' ' + ''.join(sorted(symbols_file.readline().strip('\n')))
symbols_file.close()

decode_label = lambda s: ''.join([symbols[x] for x in s[:s.index(0)]])

def decode_ctc_lite(logits, symbols):
    output, last_logit = [], None
    for logit in logits.argmax(axis=1):
        if (logit < len(symbols)) and (logit != last_logit):
            output.append(logit)
        last_logit = logit

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Model File to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--captcha-csv', help='Where to read the captchas csv to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    args = parser.parse_args()

    if args.model is None:
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
    
    interpreter = Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # read inference list

    parent_path = os.path.abspath(args.captcha_dir)

    df = pd.read_csv(args.captcha_csv, header=None, index_col=False, names=['filename'])[['filename']]
    df['result'] = ''

    start_time = time.time()
    for idx, row in df.iterrows():
        filename = row['filename']
        filename = os.path.join(parent_path, filename)

        x = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # = float64
        x = np.array(x, dtype=np.float32) / 255.0       # = float32
        x = np.expand_dims(x, axis=-1)                  # = (128, 64)
        x = x.transpose(1, 0, 2)                        # = (64, 128, 1)
        x = np.expand_dims(x, axis=0)                   # = (1, 128, 64, 1)

        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]['index'])

        y_pred = decode_label(decode_ctc_lite(y_pred[0], symbols))

        # print('%s: %s' % (os.path.basename(filename), y_pred))

        row['result'] = y_pred

        if idx % 100 == 0:
            print('predicted %d of %d images in %f seconds' % (idx, df.shape[0], time.time() - start_time))

    print('predicted %d images in %f seconds' % (df.shape[0], time.time() - start_time))
    
    df.sort_values(by=['filename'], ascending=True)
    df.to_csv(args.output, columns=['filename', 'result'], header=False, index=False)


if __name__ == '__main__':
    main()
