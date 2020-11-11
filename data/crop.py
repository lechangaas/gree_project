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
import cv2
import os
import math
import sys
import yaml
import ast
import numpy as np
sys.path.append('../')
from train.utils.yaml_dict import YAMLDict as ydict
from argparse import ArgumentParser
from preprocess import MotorDetector


if __name__ == '__main__':
    def build_argparser():
        parser = ArgumentParser()
        parser.add_argument("-i", "--input_folder", help="Path to dir:images",
                            default='data/train', type=str)
        parser.add_argument("-o", "--output_folder", help="Path to dir:images",
                            default='data/train_crop', type=str)
        parser.add_argument('-display', '--display_image', help='Display image with prediction',
                            action='store_true')

        return parser

    args = build_argparser().parse_args()
    input_folder = args.input_folder
    cfg_file = os.path.join(input_folder, 'crop-config.yaml')
    if not os.path.exists(cfg_file):
        print('config file not exist!')
        sys.exit(1)
    cfg = None
    with open(cfg_file, 'r') as stream:
        try:
            cfg = ydict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    cfg = cfg.crop_configs
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.mkdir(os.path.join(output_folder))

    input_image_folder = os.path.join(input_folder, 'image')
    input_label_folder = os.path.join(input_folder, 'label')
    output_image_folder = os.path.join(output_folder, 'image')
    if not os.path.exists(output_image_folder):
        os.mkdir(os.path.join(output_image_folder)) 
    output_label_folder = os.path.join(output_folder, 'label')
    if not os.path.exists(output_label_folder):
        os.mkdir(os.path.join(output_label_folder))
    
    detector = MotorDetector()
    for file_name in os.listdir(os.path.join(input_image_folder)):
        image_path = os.path.join(input_image_folder, file_name)
        label_path = os.path.join(input_label_folder, file_name)
        image = cv2.imread(image_path)
        label = None
        if os.path.exists(label_path):
            label = cv2.imread(label_path)
        temp_image = image.copy()
        ret, coord = detector.get_roi_simple(temp_image, cfg)
        if ret == 0:
            crop_ratio = ast.literal_eval(cfg.crop_ratio)
            suffix = 0
            for ratio in crop_ratio:
                xx, yy, rr = coord
                rr = int(rr * ratio)
                cv2.circle(temp_image, (xx, yy), rr, (0, 255, 0), 1)
                crop_image = detector.crop_roi(image, coord, ratio)
                temp_name = "%s_%s.png" % (os.path.basename(file_name).split('.')[0], suffix)
                cv2.imwrite(os.path.join(output_image_folder, temp_name), crop_image)
                if label is not None:
                    crop_label = detector.crop_roi(label, coord, ratio)
                    cv2.imwrite(os.path.join(output_label_folder, temp_name), crop_label)
                suffix += 1
                if args.display_image:
                    cv2.imshow('origin', temp_image)
                    cv2.imshow("cropped_image", crop_image)
                    if label is not None:
                        cv2.imshow("cropped_label", crop_label)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        break

        else:
            if args.display_image:
                print('File {}. Motor not found!'.format(file_name))
                cv2.imshow('origin', image)
                cv2.destroyWindow('cropped_image')
                cv2.destroyWindow('cropped_label')
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
    
