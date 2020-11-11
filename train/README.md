## training

### 1. Install dependencies

**Software requirements**

  * Ubuntu 16.04
  * Python 3.7

**Hardware requirements**

  * Intel CPU
  * Discrete GPU (for train)

**Disclaimers :** 

This code has been validated only in the following configurations:


| Hardware  | Version |
| ------------- | ------------- |
| Intel CPU  | i7-7700k or better  |
| Nvidia | 2080Ti |

  

| Software  | Version |
| ------------- | ------------- |
| Python  | 3.7  |
| Tensorflow | 1.13.1  |
| OpenCV| 4.2.0.34 |
| numpy | 1.16.4 |
| Nvidia Driver Version | 430.40 |
| CUDA | 10.1 |
| cuDNN | 7.6.4 |
| Ubuntu | 16.04 |

**Create a conda virtual environment and install build requirements :** 

```shell script
    conda create -n gree_motor python=3.7
    source activate gree_motor
    conda install tensorflow-gpu=1.13.1
    pip install -r requirements.txt
```

### 2. Training

a. **Data Prepare**

   Take the first 300 of the data set number as the training set, and the last 200 as the test set.

```shell script
    scp intel@10.67.109.92:/T2/myshare/TrainData/private/segmentation/Gree_motor/dataset_20200618.tar.gz .   
    tar -zxvf dataset_20200618.tar.gz

    python generate_csv.py -t train_folder -v test_folder
```

   Data Structure:
   
   ```
      |-data
        |---train
        |   |----image
        |   |    |-----img1.png
        |   |    |-----img2.png
        |   |    |-----img3.png
        |   |   ...
        |   |----label
        |   |    |-----img1.png
        |   |    |-----img2.png
        |   |    |-----img3.png
        |   |   ...
        |---test
        |   |----image
        |   |    |-----img4.png
        |   |    |-----img5.png
        |   |    ...
        |   |----label
        |   |    |-----img4.png
        |   |    |-----img5.png
        |   |   ...
   ```

b. **Start Training**

   The different hyper-parameters that can be used to tune the model for the dataset is extracted out in the *config.yaml* file. 

```shell script
    change train_csv_path and val_csv_path in config.yaml
    python train.py
```

### 3. Inference

a. **Evaluate Metrics**
```shell script
    python predict.py -t data/test
```

b. **Optimize using OpenVINO**

Follow instructions on OpenVINO webiste to install OpenVINO and setup the environment for Model Optimizer IR generation.

```shell script
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
      -m models/<MODEL_NAME> \
      --data_type FP32 \
      --output_dir ./models/ \
      --input_shape [1,416,416,3] \
      --model_name bin_mobilenet_fp32
```

Sample Inference

```shell script
    source /opt/intel/openvino/bin/setupvars.sh
    
    python inference.py \
     --model_file <path to OpenVino model xml file> \
     --input data/test/image \
     --output_dir <directory/where/results/are/saved> \
     --display_image <diplay image with prediction if flag is specified> \
     --device <target device for inference 'CPU, GPU, MYRIAD' - CPU is default> \
     --cpu_extension /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_avx2.so
```

> **NOTE:** You must initialize openvino environment first to use inference code.


