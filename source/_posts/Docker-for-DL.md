---
title: Docker_for_DL
mathjax: false
date: 2019-10-17 18:18:38
tags:
- docker
- dl
- ubuntu
categories: oteher
description: 为Ubuntu18.04搭建Docker深度学习环境
---
# Docker for DL in Ubuntu 18.04

[TOC]

## Install Docker

[Install Docker](https://docs.docker.com/install/),查看左侧Ubuntu安装即可。

## Install nvidia-docker

[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

1. 禁用系统默认显卡驱动

   ``` bash
   sudo nano /etc/modprobe.d/blacklist.conf
   
   # add context in the end of file
   # for nvidia display device install
   blacklist vga16fb
   blacklist nouveau
   blacklist rivafb
   blacklist rivatv
   blacklist nvidiafb
   ```

   保存后，更新内核：

   ```bash
   sudo update-initramfs -u
   ```

   

   重启电脑`sudo reboot`，然后使用`lsmod | grep nouveau`，没有输出则表示禁用成功。

2. 安装RTX 2080Ti显卡驱动
    卸载默认Nvidia驱动

   ```bash
   sudo apt-get --purge remove nvidia-*
   ```

   如果本身是图形界面，则需要进入无图形模式

   ``` bash
   sudo telinit 3
   ```

   然后使用`CTRL+ALT+F1`键，进入tty1，输入账号密码进入。

   > 此时可能遇到中文不能显示问题，可先提前将`.run`文件放置根目录，避免中文路径

   从[官网](https://www.geforce.com/drivers)查看下载2080Ti最新驱动`.run`文件，并添加运行权限:

   ```bash
   sudo chmod  +x NVIDIA-Linux-x86_64-430.50.run
   ```

   然后执行安装

   ``` bash
   sudo ./NVIDIA-Linux-x86_64-430.50.run
   ```

   第一个预安装检查会报错，点击继续安装。

   **驱动完成安装**！

   
   
   >  如果遇到下面错误
   >
   >  ``` bash
   >  WARNNING: Unable to find suitable destination to install 32-bit compatibility libraries
   >  ```
   >
   >  解决办法
   >
   >  ``` bash
   >  sudo dpkg --add-architecture i386
   >  sudo apt update
   >  ```
> sudo apt install libc6:i386
>
> ```
> 
> ```

3. 安装NVIDIA Container Toolkit

   [官网](https://github.com/NVIDIA/nvidia-docker),前提即完成Nvidia的驱动安装，不需要安装CUDA。
   
   ``` bash
   # Add the package repositories
   $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   $ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   $ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   $ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   $ sudo systemctl restart docker
   ```
   
   测试一下：
   
   ``` bash
   #### Test nvidia-smi with the latest official CUDA image
   sudo docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
   ```
   
   ```
   Thu Oct 17 10:11:06 2019       
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 430.50       Driver Version: 430.50       CUDA Version: 10.1     |
   |-------------------------------+----------------------+----------------------+
   | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |===============================+======================+======================|
   |   0  GeForce RTX 208...  Off  | 00000000:17:00.0 Off |                  N/A |
   | 27%   36C    P8     3W / 250W |      1MiB / 11019MiB |      0%      Default |
   +-------------------------------+----------------------+----------------------+
   |   1  GeForce RTX 208...  Off  | 00000000:18:00.0 Off |                  N/A |
   | 27%   39C    P8     1W / 250W |      1MiB / 11019MiB |      0%      Default |
   +-------------------------------+----------------------+----------------------+
   |   2  GeForce RTX 208...  Off  | 00000000:65:00.0  On |                  N/A |
   | 29%   43C    P8    21W / 250W |    559MiB / 11016MiB |      0%      Default |
   +-------------------------------+----------------------+----------------------+
   |   3  GeForce RTX 208...  Off  | 00000000:B3:00.0 Off |                  N/A |
   | 27%   33C    P8     3W / 250W |      1MiB / 11019MiB |      0%      Default |
   +-------------------------------+----------------------+----------------------+
                                                                                  
   +-----------------------------------------------------------------------------+
   | Processes:                                                       GPU Memory |
   |  GPU       PID   Type   Process name                             Usage      |
   |=============================================================================|
   +-----------------------------------------------------------------------------+
   ```
   
   

## Use Docker for DL

## 创建数据卷

[参考](https://docs.docker.com/engine/reference/commandline/volume_create/)

需要区分下type的区别 (不使用此方法，卷会默认于docker目录而不是想要的目录)

``` bash
docker volume create -d local \
--opt type=tmpfs \
--opt device=/home/chen/data/ \
chen_data
```



way2:

首先使用账户`chen`创建所需要挂载的目录(即保证目录的owner为chen, docker自行创建的为root)

然后直接使用 mount 参数进行挂载即可



## 运行image

使用[deepo](https://github.com/ufoym/deepo)

``` bash
docker run \
 --name chenshuai \
 --gpus all \
 -it \
 -d \
 -p 8888:8888 \
 --ipc=host \
 --mount type=bind,source=/home/chen/docker_env,target=/env \
 ufoym/deepo:all-jupyter-py36
```

> gpus 表示使用所有GPU
>
> it 表示ternimal
>
> d 表示守护式启动
>
> p 表示端口映射
>
> -- mount 添加挂载

## 查看所有container

``` bash
docker container ls
```

## 进入container

``` bash
docker exec -it <PID> bash
```

## 删除未运行container

``` bash
docker container prune -f
```



## install docker-compose

[参考](curl -L https://raw.githubusercontent.com/docker/compose/1.24.1/contrib/completion/bash/docker-compose > /etc/bash_completion.d/docker-compos)

使用pip安装

```bash
sudo pip install -U docker-compose
```

bash 补全命令

```bash
curl -L https://raw.githubusercontent.com/docker/compose/1.24.1/contrib/completion/bash/docker-compose > /etc/bash_completion.d/docker-compose
```



## install docker-registry

pass



## pycharm remote connect docker(GPU)

首先运行docker并暴露22端口

```bash
docker run \
 --name zhannan \
 --gpus all \
 -it \
 -d \
 -p 9188:8888 \
 -p 9122:22 \
 --ipc=host \
 --mount type=bind,source=/home/zhang/docker_env,target=/env \
 ufoym/deepo:all-jupyter-py36
```

进入容器后，开启root的远程SSH连接

```bash
apt update && apt install -y openssh-server && apt install -y nano
```

修改`/etc/ssh/sshd_config`文件，添加

```
PermitRootLogin yes
```

重启SSH服务

```bash
service ssh restart
```

设置`root`密码后，验证连接

```bash
# 容器内使用22端口
ssh root@127.0.0.1
# 容器外使用映射端口
ssh root@127.0.0.1 -p 9122
```

添加`pycharm`的远程连接