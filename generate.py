#!/usr/bin/env python3

import os
import numpy
import random
import string
import random
import cv2
import argparse
import captcha.image

symbols_file = open('symbols.txt', 'r')
symbols = ' ' + ''.join(sorted(symbols_file.readline().strip('\n')))
symbols_file.close()

lens = list(range(1, 7))
max_len = 8

height, width = 64, 128

decode_label = lambda s: ''.join([symbols[x] for x in s[:s.index(0)]])
encode_label = lambda s: [symbols.find(x) for x in s.ljust(max_len)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    args = parser.parse_args()

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    captcha_generator = captcha.image.ImageCaptcha(width, height)

    print("Generating captchas with symbol set {" + symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    for _ in range(args.count):

        random_str = ''.join([random.choice(symbols[1:]) for _ in range(random.choice(lens))])
        image_path = os.path.join(args.output_dir, random_str+'.png')
        if os.path.exists(image_path):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, random_str + '_' + str(version) + '.png')):
                version += 1
            image_path = os.path.join(args.output_dir, random_str + '_' + str(version) + '.png')

        image = numpy.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)

if __name__ == '__main__':
    main()
