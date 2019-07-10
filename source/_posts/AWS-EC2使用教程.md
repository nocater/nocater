---
title: AWS_EC2使用教程
mathjax: false
date: 2019-03-21 19:19:48
tags: 
- AWS_EC2
categories: other
description: 关于建立AWS EC2服务器的创建方式
---
既可以选择使用一年的AWS免费服务器，也可以选择购买GPU服务器。
# 注册AWS
首先注册AWS，需要一张外币信用卡(Master卡可行)。

# 选择服务器地区及版本
在[Amazon EC2 定价](https://aws.amazon.com/cn/ec2/pricing/on-demand/)查看各地区版本价格。一定要先确定服务器地区及版本，不同地区的相同版本价格是不一样的。以`p2.xlarge`为例，美国等地区基本是0.9$/hour，但是亚洲地区的是在1.1$起步。

# 更改上限
在选择好地区及版本后，还需要去[EC2 Service Limit report](https://console.aws.amazon.com/ec2/v2/home?#Limits)配置一下示例的运行上限，此步骤是不需要钱的。如果是先选择了购买，在最后一步可能会提示：
```
Launch Failed
You have requested more instances (1) than your current instance limit of 0 allows for the specified instance type. Please visit http://aws.amazon.com/contact-us/ec2-request to request an adjustment to this limit.
Hide launch log
Initiating launches
 
FailureRetry
```
在[Support Center](http://aws.amazon.com/contact-us/ec2-request)提交修改limit申请即可，需要十分钟及4 5封邮件时间。

# 购买实例
访问[EC2 Management Console](https://console.aws.amazon.com/ec2/v2/home)选择 `Launch Instance`。

在左侧选择`AWS Marketplace`，输入`deep learning ubuntu`进行搜索，选择`Deep Learning AMI(Ubuntu)`。

在价格详细页面选择`Continue`。

选择已经决定的服务器配置类型。

`Configure Instance(配置实例)`、`Add Storage(添加存储)`、`Add Tags(添加标签)`这几步俊保留默认配置，在`Configure Security Group(配置安全组)`需要自定义配置。创建一个自定义的TCP规则来允许8888端口，并选择`Anywhere`
来允许任何IP访问。

在创建示例过程最后，系统会询问你想要创建新的连接密钥还是重复使用现有的，如果没有使用过创建一个新的就可以了。

# 运行实例
Window可以使用Git的终端进行操作。

使用SSH连接实例,请自行替换部分内容：  
```
cd /Users/your_username/Downloads/

chmod 0400 <your .pem filename>  

ssh -L localhost:8888:localhost:8888 -i <your .pem filename> ubuntu@<your instance DNS>
```

在终端中，使用`jupyter notebook`启动程序，然后复制打开链接。

# 终止实例

返回到[AWS Management Console ](https://console.aws.amazon.com/)，选择实例，选择`Action`，找到`Instance State, `，单击`Terminate`。

# 参考
1. [Launch an AWS Deep Learning AMI with Amazon EC2](https://aws.amazon.com/cn/getting-started/tutorials/get-started-dlami/?trk=gs_card)
2. [Amazon EC2 定价](https://aws.amazon.com/cn/ec2/pricing/on-demand/)
3. [DeepLearning笔记：如何用亚马逊云服务 GPU 训练神经网络](https://www.uegeek.com/180322-DeepLearning11-aws-gpu-training.html)