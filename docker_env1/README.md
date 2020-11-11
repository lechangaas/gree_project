# Build training docker images
## Build intel avx512 environment
备注：默认安装python3.6, tensorflow1.15.2
* 首先拉取镜像：intel/intel-optimized-tensorflow:1.15.2
* 执行脚本：bash build_avx512_env.sh

## Build nvidia gpu environment
备注：默认安装python3.7, tensorflow1.13.1
* 首先拉取镜像：nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04  
拉取的镜像版本要和显卡驱动相适配，当前测试过的显卡驱动为430.40，所以适配cuda10 cudnn7  
如果更改了镜像版本，同时要修改Dockerfile_gpu里的第一句话为新的镜像名
* 安装nvidia docker环境
```
sudo apt-get purge nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
注意：如果curl有报错，可能要过代理 ，使用-x参数, 例如：-x http://child-prc.intel.com:913
* 执行脚本：bash build_gpu_env.sh

# Using training docker images
## Using intel avx512 environment
docker run --rm -it cpu_avx512:0.1  
备注：如果需要挂载卷，则在启动容器时要添加-v参数，例：-v /home/dls1/Desktop/docker_test:/home/training

## Using nvidia gpu environment
docker run --rm --runtime=nvidia -it tf13_gpu:0.1  
备注：如果需要挂载卷，则在启动容器时要添加-v参数，例：-v /home/dls1/Desktop/docker_test:/home/training