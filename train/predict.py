"""
[INTEL CONFIDENTIAL]

Copyright (c) 2019 Intel Corporation.

This software and the related documents are Intel copyrighted materials, and
your use of them is governed by the express license under which they were
provided to you ("License"). Unless the License provides otherwise, you may
not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express
or implied warranties, other than those that are expressly stated in the License.
"""

import os
import cv2
import sys
import yaml
import random
import glob
import ast

import numpy as np
from argparse import ArgumentParser
from utils.yaml_dict import YAMLDict as ydict
from utils.model import build_model

# Read config file
cfg = None
with open('config.yaml', 'r') as stream:
    try:
        cfg = ydict(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(1)
cfg = cfg.base_model_configs

# Set visible GPU devices
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu


parser = ArgumentParser()
parser.add_argument("-t", "--test_dir", help="test folder", default='../data/test_crop', type=str)
parser.add_argument("-m", "--model_dir", help="model folder", default='models', type=str)
parser.add_argument("-th", "--threshold", help="inference threshold", default=0.5, type=float)
parser.add_argument('-dis', '--display_image', help='Display image with prediction', action='store_true')
args = parser.parse_args()


def apply_mask(image, mask, color, alpha=1):
    for c in range(3):
        image[:, :, c] = np.where(mask > 0,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


if __name__ == '__main__':
    input_folder = args.test_dir
    model_dir = args.model_dir
    threshold = args.threshold
    image_dir = os.path.join(args.test_dir, 'image')
    result_dir = os.path.join(args.test_dir, 'result')

    # Parameter check
    if not os.path.exists(image_dir):
        print("image_dir: %s doesn't exit" % image_dir)
        sys.exit(0)
    if not os.path.exists(model_dir):
        print("model_path: %s doesn't exit" % model_dir)
        sys.exit(0)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    names = glob.glob(model_dir + '/' + '*.hdf5')
    if len(names) is None:
        print("model_path: model file doesn't exit")
        sys.exit(0)
    names.sort(key=lambda fn: os.path.getmtime(fn))
    print("%d model(s), model averaging will be done if there are one more models" % len(names))

    model, preprocess_input = build_model(cfg)
    model.load_weights(names[-1])

    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))
        print(os.path.join(image_dir, image_name))
        image_bgr = image.copy()
        image = cv2.resize(image,  (cfg.img_width, cfg.img_height))
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        result = np.squeeze(model.predict(image))
        result[result < threshold] = 0
        result = cv2.resize(result, (image_bgr.shape[1], image_bgr.shape[0]))
        mask = apply_mask(image_bgr, result, [0, 0, 255], 0.5)
        cv2.imwrite(os.path.join(result_dir, image_name), mask)
        if args.display_image:
            cv2.namedWindow('inference_result', cv2.WINDOW_NORMAL)
            cv2.putText(mask, "Press 'q' to exit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [128, 128, 240], 2)
            cv2.imshow('inference_result', mask)
            key = cv2.waitKey(0) & 0xff
            if key == ord('q'):
                break


