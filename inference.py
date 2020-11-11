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
import cv2
import random
import numpy as np
import logging
from openvino.inference_engine import IENetwork, IECore

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)

class InferenceEngine:

    def __init__(self,
                 model_xml,
                 device='CPU',
                 cpu_extension=None):

        # Parameter check
        self.model_xml = model_xml
        self.model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Plugin initialization for specified device and load extensions library if specified
        logging.info("Creating Inference Engine")
        logging.info("Device info:")
        print("{}{}".format(" " * 8, device))
        self.ie = self._prepare_plugin(cpu_extension, device)
        versions = self.ie.get_versions(device)
        print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[device].major,
                                                              versions[device].minor))
        print("{}Build ........... {}".format(" " * 8, versions[device].build_number))

        # Read IR
        self.net = self._load_network()
        self._check_supported_layers(device)

        # Preparing input blobs
        logging.info("Preparing input blobs")
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

        # Loading model to the Plugin
        self.net.batch_size = 1
        logging.info("Loading model to the plugin")
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)

        # Read and pre-process input images
        self.input_shape = self.net.inputs[self.input_blob].shape
        del self.net

    def _prepare_plugin(self, cpu_extension, device):
        logging.info("Loading Inference Engine")
        ie = IECore()
        if cpu_extension and "CPU" in device:
            ie.add_extension(cpu_extension, "CPU")
            logging.info("CPU extension loaded: {}".format(cpu_extension))

        return ie

    def _load_network(self):
        logging.info("Loading network files:\n\t{}\n\t{}".format(self.model_xml, self.model_bin))
        return IENetwork(model=self.model_xml, weights=self.model_bin)
        # return self.ie.read_network(model=self.model_xml, weights=self.model_bin)

    def _check_supported_layers(self, device):
        """
        Check if network layers are supported by OpenVino
        :param net: Read network
        :return: None
        """
        if "CPU" in device:
            supported_layers = self.ie.query_network(self.net, "CPU")
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                logging.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                              format(device, ', '.join(not_supported_layers)))
                logging.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)

    def _normalizer(self, x, mode='tf'):
        x = x.astype(np.float32)

        if mode == 'tf':
            x /= 127.5
            x -= 1.
        elif mode == 'caffe':
            x -= [103.939, 116.779, 123.68]
        elif mode == 'torch':
            x /= 255.
            x -= [0.485, 0.456, 0.406]
            x /= [0.229, 0.224, 0.225]

        return x

    def _preprocess_image(self, image):
        '''
         Read and pre-process input image
        :param image:
        :return: processed image
        '''
        _, _, ih, iw = self.input_shape
        if image.shape[:-1] != (ih, iw):
            image = cv2.resize(image, (iw, ih))
        image = self._normalizer(image)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)

        return image

    def start_infer(self, image):
        '''
         Start sync inference
        :param image:
        :return: inference result
        '''
        processed_img = self._preprocess_image(image)
        return np.squeeze(self.exec_net.infer(inputs={self.input_blob: processed_img})[self.out_blob])

'''
if __name__ == '__main__':
    xml = '2019_IR\\bin_mobilenet_2020-06-17_fp32.xml'
    data_path = 'image'

    engine = InferenceEngine(xml, device='CPU')
    image = cv2.imread(os.path.join(data_path, random.choice(os.listdir(data_path))))

    res = engine.start_infer(image)

    cv2.imshow('s', res)
    cv2.waitKey(0)
'''