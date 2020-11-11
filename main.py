"""
[INTEL CONFIDENTIAL]

Copyright (c) 2020 Intel Corporation.

This software and the related documents are Intel copyrighted materials, and
your use of them is governed by the express license under which they were
provided to you ("License"). Unless the License provides otherwise, you may
not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express
or implied warranties, other than those that are expressly stated in the License.
"""

import sys
import os
import time
import glob
import logging
import yaml
import cv2
import ast
import numpy as np
from openvino.inference_engine import IENetwork, IECore
from train.utils.yaml_dict import YAMLDict as ydict
from argparse import ArgumentParser
from preprocess import MotorDetector
from inference import InferenceEngine


def inference_resize(exec_net, input_blob, out_blob, input_image):
    input_image = np.asarray(input_image, dtype=np.float32)
    input_image /= 127.5
    input_image -= 1.
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.reshape(input_image, (1,) + input_image.shape)

    time_start = time.time()
    res = exec_net.infer(inputs={input_blob: input_image})
    time_end = time.time()
    total_time = (round(time_end - time_start, 3)) * 1000
    logging.info('1/1 inference time - {}ms/step'.format(total_time))
    res = np.squeeze(res[out_blob])

    return res


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask > 0,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


if __name__ == '__main__':
    def build_argparser():
        parser = ArgumentParser()
        parser.add_argument("-i", "--input_folder", help="Path to dir:images",
                            default='data/test/image', type=str)
        parser.add_argument("-m", "--model_dir", help="model folder", default='model', type=str)
        parser.add_argument("-d", "--device",
                            help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD "
                                 "is acceptable (CPU by default)", default="CPU", type=str)
        parser.add_argument("-t", "--thresh", help="segmentation pixel threshold", default=0.5, type=float)
        parser.add_argument("-o", "--output_folder", help="Path to dir:images",
                            default='data/test/result', type=str)
        parser.add_argument('-dis', '--display_image', help='Display image with prediction',
                            action='store_true')

        return parser


    args = build_argparser().parse_args()
    input_folder = args.input_folder
    cfg_file = os.path.join(os.path.dirname(input_folder), 'crop-config.yaml')
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

    image_dir = os.path.join(input_folder)
    detector = MotorDetector()
    model_dir = args.model_dir
    infer_device = args.device
    thresh = args.thresh

    # Parameter check
    if not os.path.exists(image_dir):
        print("image_dir: %s doesn't exit" % image_dir)
        sys.exit(0)
    if not os.path.exists(model_dir):
        print("model_path: %s doesn't exit" % model_dir)
        sys.exit(0)
    models = glob.glob(model_dir + '/' + '*.xml')
    if len(models) is None:
        print("model_path: model file doesn't exit")
        sys.exit(0)
    model_xml = models[0]
    print('load model from: {}'.format(model_xml))

    engine = InferenceEngine(model_xml, device=infer_device)
    if args.display_image:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    for name in os.listdir(image_dir):
        image_name = os.path.join(image_dir, name)
        image = cv2.imread(image_name)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        temp_image = image.copy()
        ret, coord = detector.get_roi_simple(temp_image, cfg)
        height, width, _ = image.shape
        if ret == 0:
            crop_ratio = ast.literal_eval(cfg.crop_ratio)
            for ratio in crop_ratio:
                xx, yy, r = coord
                rr = int(r * ratio)
                crop_image = detector.crop_roi(image, coord, ratio)
                cut_coor, pad_coor = detector.calculate_coor(height, width, xx, yy, rr)

                init_h, init_w = crop_image.shape
                result = engine.start_infer(cv2.cvtColor(crop_image, cv2.COLOR_GRAY2RGB))
                result[result < thresh] = 0
                result[result >= thresh] = 1
                result = result.astype(np.uint8)
                result = cv2.resize(result, (init_w, init_h))

                # find holes
                pad_xmin, pad_ymin, pad_xmax, pad_ymax = pad_coor
                image_for_hole = crop_image * result
                cv2.circle(image_for_hole, (init_w // 2, init_h // 2), int(r * 0.9), 255, -1)
                detect_hole_image = image_for_hole[pad_ymin:pad_ymax, pad_xmin:pad_xmax]  # cut padded image
                hole_circles = detector.find_holes(detect_hole_image, cfg)

                # visualize
                cut_xmin, cut_ymin, cut_xmax, cut_ymax = cut_coor
                infer_mask = np.zeros((height, width), dtype=np.uint8)
                infer_mask[cut_ymin:cut_ymax, cut_xmin:cut_xmax] = result[pad_ymin:pad_ymax, pad_xmin:pad_xmax]
                mask = apply_mask(image, infer_mask, [0, 0, 255])
                cv2.circle(image, (xx, yy), rr, (0, 255, 0), 2)
                cv2.circle(image, (xx, yy), r, (255, 0, 0), 2)
                if hole_circles is not None:
                    hole_circles = np.int0(hole_circles)
                    hole_xs = hole_circles[:, 0] + cut_xmin
                    hole_ys = hole_circles[:, 1] + cut_ymin
                    hole_rs = hole_circles[:, 2]
                    for i in range(len(hole_xs)):
                        cv2.circle(image, (hole_xs[i], hole_ys[i]), hole_rs[i], (255, 0, 0), 2)
                    cv2.putText(image, 'screw holes:%d' % len(hole_circles), (5, 50),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)

            cv2.putText(image, name, (5, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(output_folder, name), image)
            if args.display_image:
                cv2.imshow('image', image)
                key = cv2.waitKey(0) & 0xff
                if key == ord('q'):
                    break
