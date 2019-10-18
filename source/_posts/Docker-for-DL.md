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

# Reference

[How to set up Docker for Deep Learning with Ubuntu 16.04 (with GPU)](<https://johannesfilter.com/how-to-set-ubuntu-16-04-for-deep-learning-with-gpu/>)

[dl-setup](<https://github.com/floydhub/dl-setup#nvidia-drivers>)

[Ubuntu18.04上安装RTX 2080Ti显卡驱动](<https://my.oschina.net/u/2306127/blog/2877804>)

[Ubuntu 18.04 安装 NVIDIA 显卡驱动](<https://zhuanlan.zhihu.com/p/59618999>)