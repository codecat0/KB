---
Date: 2026-06-10
tags:
  - 分割
  - 腹部多器官
来源: arXiv
年份: "2024"
机构: 云南大学
Code: https://github.com/gndlwch2w/msvm-unet
---

## **1. 论文信息**

状态空间模型（SSMs）特别是Mamba模型能以线性复杂度建模长距离依赖，适合医学图像分割。但准确分割需多尺度特征与全局依赖，**现有 CNN 和 SSMs 整合工作未解决多尺度特征捕获及方向敏感性问题。为此提出 MSVM - UNet，在 VSS 块引入多尺度卷积以捕获聚合多尺度特征，用大核块扩展（LKPE）层高效上采样**。Synapse 和 ACDC 数据集实验显示其优于部分先进方法。

## **2. 本文贡献**

本文提出了一种新型的多尺度视觉Mamba UNet（MSVM-UNet）模型，用于医学图像分割，具有以下三个主要创新点：

- **多尺度视觉状态空间（MSVSS）块**：作者提出了MSVSS块，它结合了交叉扫描模块（CSM）和多尺度卷积操作，这不仅有效地模拟了像素之间的长距离依赖关系，还捕获了多尺度特征表示。这种设计使得模型能够同时处理不同尺寸和形状的物体，并更好地定位器官边界。

- **大核块扩展（LKPE）层**：为了解决传统上采样方法在空间关系整合方面的不足，作者引入了LKPE层。该层通过在扩展通道维度之前引入大核深度卷积，实现了特征图的更高效上采样，同时聚合了通道和空间信息，从而获得了更具判别力的特征表示。

- **多尺度卷积核的选择**：在MSVSS块中，作者探索了不同多尺度卷积核的组合，并选择了[1, 3, 5]作为默认的多尺度卷积核。这种选择在避免过度计算开销的同时，优化了模型的性能。

## **3. 创新方法**

[![](https://pic1.imgdb.cn/item/67fccb8c88c538a9b5d0bbb6.png)](https://pic1.imgdb.cn/item/67fccb8c88c538a9b5d0bbb6.png)

MSVM-UNet采用了一个U型分层次编码器-解码器架构，并在编码器与解码器之间采用了short-cut连接。

- 编码器采用由ImageNet-1k数据集预训练的VMamba V2，其包含4个stage，第一个stage由Patch Embedding模块和VSS块组成，剩下3个stage均由负责下采样的Patch Merging模块和VSS块组成；

- 解码器包含3个stage和FLKPE输出层构成，其中每个stage均由负责上采样的LKPE模块和MSVSS块组成，MSVSS模块的输入来自编码器VSS块的输出和来自负责上采样的LKPE模块的输出的concat结果；

### **3.1 Multi-Scale Vision State Space (MSVSS) Block**

MSVSS块通过在VSS块中引入多尺度前馈网络（MS-FFN）来解决同时捕获多尺度详细特征和在二维视觉数据中有效地解决方向敏感性问题。

- 二维选择扫描块（SS2DBlock）model了每个特征在四个方向上的长程依赖性；

- MS-FFN中的卷积操作从四个剩余的对角方向聚合信息，以增强特征表示；

- 为了有效地捕获和聚合多尺度特征表示，MSVSS 采用了一组具有不同核大小并行的卷积操作来实现这一目标；

[![](https://pic1.imgdb.cn/item/67fcceb188c538a9b5d0bdfb.png)](https://pic1.imgdb.cn/item/67fcceb188c538a9b5d0bdfb.png)

### 2D-Selective-Scan Block (SS2DBlock)

SS2D首先将2D输入特征图沿着四个不同的扫描路径进行 flatten，得到四个一维序列。这些序列随后被输入S6块进行选择性扫描，以模拟长程依赖关系。最后，将四个一维序列恢复为原始2D形式并将它们相加以产生输出。

### Multi-Scale Feed-Forward Neural Network (MS-FFN)

MSFNN中引入了卷积操作来聚合这四个对角方向的信息，并采用了一组**使用不同 Kernel 大小的卷积操作用于有效地捕捉层叠特征的多细节信息和高分辨率特征表示**

### **3.2 Large Kernel Patch Expanding (LKPE) Layer**

[![](https://pic1.imgdb.cn/item/67fccffb88c538a9b5d0bf8a.png)](https://pic1.imgdb.cn/item/67fccffb88c538a9b5d0bf8a.png)

Patch Expanding层仅考虑特征的通道信息，而忽略了相邻特征之间的空间关系，为解决此问题，LKPE首先应用一个卷积将通道维数翻倍，然后进行批量归一化和ReLU激活函数，接着使用有效卷积聚合空间信息，并最终通过扩展包含空间和通道信息的特征表示进行上采样

## **4. 实验分析**

### **4.1 实验细节**

- 在NVIDIA GeForce RTX 3090基于Pytorch框架实现

- 数据增强：将图像Resize至224大小、水平翻转、垂直翻转、随机旋转、高斯噪声、高斯模糊和对比增强；

- Batch Size：32

- 优化器：AdamW

- 迭代次数：300

- 学习率：初始学习率为5e-4，并采用余弦衰减策略

- 损失函数：Dice Loss和CE Loss的组合

### **4.2 定量比较**

在腹部多器官数据集（Synapse）上训练的各种医学图像分割方法的定量比较中，MSVM-UNet在多个指标均超过其他基线方法。

[![](https://pic1.imgdb.cn/item/67fcd31d88c538a9b5d0c15b.png)](https://pic1.imgdb.cn/item/67fcd31d88c538a9b5d0c15b.png)

### **4.3 定性比较**

在腹部多器官数据集（Synapse）上训练的各种医学图像分割方法的定性比较中，MSVM-UNet不仅能有效处理形状和大小各异的器官，还能更好地定位器官边界。

[![](https://pic1.imgdb.cn/item/67fcd46788c538a9b5d0c27b.png)](https://pic1.imgdb.cn/item/67fcd46788c538a9b5d0c27b.png)

### **4.4 消融研究**

- 在腹部多器官数据集（Synapse）上消融实验表明：**MSVSS块与LKPE能提高分割Dice指标**；

- 对不同上采样方式进行消融实验表明：**LKPE效果最优**；

- 对不同大小卷积核进行消融实验表明：**卷积核为1、3、5时，多尺度特征提取模块效果最优**；

[![](https://pic1.imgdb.cn/item/67fcd4e388c538a9b5d0c364.png)](https://pic1.imgdb.cn/item/67fcd4e388c538a9b5d0c364.png)

[![](https://pic1.imgdb.cn/item/67fcd51288c538a9b5d0c3a2.png)](https://pic1.imgdb.cn/item/67fcd51288c538a9b5d0c3a2.png)

## **5. 结论**

- 本文提出了一种新颖的多尺度视觉Mamba UNet，旨在解决医学图像分割面临的挑战。由于多尺度深度卷积的设计，MSVM-UNet不仅能捕获不同尺度下的信息并建模所有方向上的长程依赖性，还能保持计算效率和可接受的参数数量；

- 通过有效集成通道和空间信息进行上采样，MSVM-UNet实现了更具有判别性的特征表示，从而使得医学图像分割的结果更加准确；