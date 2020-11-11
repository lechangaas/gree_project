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

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder

# Limit user access to only home directory of the user
safe_dir = str(Path.home())

def is_safe_path(input_dir):
    if not os.path.isdir(input_dir):
        raise argparse.ArgumentTypeError("Specified directory does not exist.")
    if not os.path.abspath(input_dir).startswith(safe_dir):
        raise argparse.ArgumentTypeError("Specified directory is in an unsafe path.\n" + 
            "Move directory within safe loc {} to access.".format(safe_dir))
    return input_dir

def build_argparser():
    parser = argparse.ArgumentParser()
    # Add argument for dataset dir path
    parser.add_argument('--train', '-t', default = 'data/train', #type = is_safe_path,
                    help = 'Path to train data dir')
    parser.add_argument('--val', '-v', default = 'data/val', #type = is_safe_path,
                    help = 'Path to val data dir')

    return parser
 
def main():
    args = build_argparser().parse_args()
    
    # Put folder names in a list, these will be the raw_labels
    train_image_folders = os.path.join(args.train, 'image')
    train_label_folders = os.path.join(args.train, 'label')
    val_image_folders = os.path.join(args.val, 'image')
    val_lable_folders = os.path.join(args.val, 'label')

    # Create lists for train/val/test images and raw_labels
    train_image_list = []
    train_raw_label_list = []
    val_image_list = []
    val_raw_label_list = []
    
    # Grab image in each train folder
    img_in_folder = os.listdir(train_image_folders)
	
    # Add each image and its raw_label to respective list
    for i in img_in_folder:
        train_image_list.append(os.path.realpath(os.path.join(train_image_folders, i)))
        train_raw_label_list.append(os.path.realpath(os.path.join(train_label_folders, i)))
    
    img_in_folder = os.listdir(val_image_folders)
        
    for i in img_in_folder:
        val_image_list.append(os.path.realpath(os.path.join(val_image_folders, i)))
        val_raw_label_list.append(os.path.realpath(os.path.join(val_lable_folders, i)))
   
    # Create dataframe
    train_df = pd.DataFrame(OrderedDict((('images', train_image_list),
                                        ('label', train_raw_label_list))))
    
    val_df = pd.DataFrame(OrderedDict((('images', val_image_list),
                                      ('label', val_raw_label_list))))
    
   
    # Save to csv file
    train_csv = os.path.join(args.train, 'train.csv')
    val_csv = os.path.join(args.val, 'val.csv')
    train_df.to_csv(train_csv)
    val_df.to_csv(val_csv)
    
    print('Train, val CSVs generated')
    print('----------------------------')
    print('Train CSV : {}'.format(train_csv))
    print('Val CSV : {}'.format(val_csv))


if __name__ == '__main__':
    main()
