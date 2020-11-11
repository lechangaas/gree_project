# Gree Motor Pipeline

## Requirements
### use local environment
**Software requirements**

  * Ubuntu 
  * Anaconda 
  * OpenVINO 

**Hardware requirements**

  * Intel CPU
  * Discrete GPU (for train)

**Disclaimers :** 

This code has been validated only in the following configurations:


| Hardware  | Version |
| ------------- | ------------- |
| Intel CPU  | i5-6200 or better  |
| Nvidia | 2080Ti |

  
| Software  | Version |
| ------------- | ------------- |
| Anaconda  | Python3.7  |
| Nvidia Driver Version | 430.40 |
| CUDA | 10.1 |
| cuDNN | 7.6.4 |
| Ubuntu | 16.04 |
| OpenVINO | 2020.3.194 |

**Dependencies :** 
```bash
 $ conda install tensorflow-gpu=1.13.1
 $ pip install -r requirements.txt
```
### use docker environment
see docker_env

## Data Prepare


**Data Structure**

There is a `crop-config.yaml` under each data folder, which is used to specifies different tunable and hyper parameters of cropping.

```
      |-data
        |---train_0
        |   |----image
        |   |    |-----img1.jpg
        |   |   ...
        |   |----label
        |   |    |-----img1.jpg
        |   |   ...
        |   |----crop-config.yaml
        |   | 
        |---train_1
        |   |----image
        |   |    |-----img2.jpg
        |   |   ...
        |   |----label
        |   |    |-----img2.jpg
        |   |   ...
        |   |----crop-config.yaml
        |   | 
        |---train_2
        |   |----image
        |   |    |-----img2.jpg
        |   |   ...
        |   |----label
        |   |    |-----img2.jpg
        |   |   ...
        |   |----crop-config.yaml
        |   | 
        |---val
        |   |----image
        |   |    |-----img3.jpg
        |   |    ...
        |   |----label
        |   |    |-----img3.jpg
        |   |   ...
        |   |----crop-config.yaml
        |   | 
        |---test
        |   |----image
        |   |    |-----img4.jpg
        |   |    |-----img5.jpg
        |   |    ...
        |___|----crop-config.yaml
   ```

**Crop the ROI from the original dataset**

The function finds circles in a grayscale image using a modification of the Hough transform, and different hyper-parameters can be adjusted in the `crop-config.yaml` file.
> **NOTE:** \
> Usually the function detects the centers of circles well. However, it may fail to find correct radii. You can assist to the function by specifying the radius range ( **minRadius** and **maxRadius** ).

| Parameters  |  |
| ------------- | ------------- |
| dp  | Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.   |
| minDist | Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed. |
| param1 | 	First method-specific parameter. It is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).  |
| param2 | Second method-specific parameter. It is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.  |
| minRadius | Minimum circle radius. |
| maxRadius | Maximum circle radius. If <= 0, uses the maximum image dimension. If < 0, returns centers without finding the radius. |

```shell script
    $ cd data
    $ python crop.py -i train_0 -o train_crop
    $ python crop.py -i train_1 -o train_crop
    $ python crop.py -i train_2 -o train_crop
    $ python crop.py -i train_3 -o train_crop
    $ python crop.py -i val -o val_crop
```

## Training
**Train the model on the cropped dataset**

`train/config.yaml` specifies different tunable and hyper parameters that are in use for training the binary model. They are extracted out of the training code to allow for easy tuning and testing.
```shell script
    $ cd ../train
    $ python generate_csv.py -t ../data/train_crop -v ../data/val_crop
    $ python train.py
```

**Test the model to the cropped test data**
```shell script
    $ python predict.py -t ../data/test_crop -display
```

## Inference
------------

**Convert the model to OpenVINO format**
```shell script
    $ source /opt/intel/openvino_2020.3.194/bin/setupvars.sh
    $ python /opt/intel/openvino_2020.3.194/deployment_tools/model_optimizer/mo_tf.py \
          -m models/<PB FIlE> \
          --data_type FP32 \
          --output_dir ../model/ \
          --input_shape [1,416,416,3] \
          --model_name bin_mobilenet_fp32
```

**Run the whole pipeline, which includes the whole Crop ROI, Inference, Screw Hole Detection**
```shell script
    $ cd ../
    $ python main.py -i data/test/image -m model -dis
```

## Test Accuracy
**Test accuracy of finding holes**  
make `label.txt` like this:
```
7000.jpg 5
7002.jpg 3
7003.jpg 6
7004.jpg 4
...
```
run script:
```shell script
    $ python main_acc.py -i data/test/image -m model -l data/test/label.txt -o data/test/result
```
The result will save in data/test/result


