---
Date: 2026-06-09
tags:
  - object_detection
来源: AAAI
年份: "2024"
机构: 西南科技大学、南京理工大学
Code: https://github.com/JN-Yang/PConv-SDloss-Data
---
## 一、简述
### 1. 研究背景

- **研究问题**：红外小目标检测和分割（IRSTDS）在军事和民用领域具有重要应用，但现有基于卷积神经网络（CNN）的方法通常使用标准卷积，未充分考虑红外小目标像素分布的空间特性。此外，现有损失函数未能充分考虑不同目标尺度下尺度和位置损失的敏感性差异，限制了对暗小目标的检测性能。
- **研究难点**：红外小目标通常因距离远而呈现暗淡、低信噪比（SNR）和低信号杂波比（SCR），缺乏纹理信息，且目标大小和形状随距离变化，复杂背景进一步遮蔽目标。现有数据集存在小目标比例低、背景简单、数据规模小等问题，限制了检测器在复杂现实场景中的性能。
- **文献综述**：传统模型驱动方法依赖于先验知识的手动参数调整，适应性差，鲁棒性低。相比之下，基于数据驱动的深度学习（DL）方法利用大量和多样化的IRSTDS数据，通过损失函数的梯度下降实现参数的自动更新，具有更强的鲁棒性。CNN基础的IRSTDS方法主要分为基于检测的方法和基于分割的方法。现有损失函数如GIoU、CloU、NWD、SAFit等在处理IoU波动误差和不同尺度目标的敏感性方面存在局限性。

### 2. 本文贡献

- **PConv模块设计**：提出了一种新颖的pinwheel-shaped convolution（PConv）模块，该模块通过不对称填充创建水平和垂直方向的卷积核，以适应红外小目标的高斯空间分布特性。PConv模块在骨干网络的较低层替代标准卷积，以增强特征提取能力，显著增加感受野，并且只引入了最小的参数增加。
- **感受野与参数效率**：PConv模块通过分组卷积显著扩大了感受野，同时最小化了参数数量的增加。例如，PConv(3,3)相较于3×3标准卷积，感受野增加了177%，参数仅增加了111%。PConv(4,3)的感受野增加了444%，参数仅增加了122%。通过将PConv和标准卷积的输出结果进行对比，展示了PConv在增强红外小目标与背景对比度的同时，抑制了杂乱信号。

## 二、方法

主要分为两个核心创新部分：**Pinwheel-shaped Convolution (PConv，风车形卷积)** 和 **Scale-based Dynamic (SD) Loss（基于尺度的动态损失）**。这两个模块分别针对红外小目标（IRST）的空间分布特性和标签标注不确定性（尤其是小目标的IoU波动）进行优化，可即插即用，适用于检测（BBox）和分割（Mask）任务。

### 1. Pinwheel-shaped Convolution (PConv，风车形卷积)

**设计动机**：红外小目标在图像中呈现高斯（Gaussian）空间分布（中心亮度高，向外快速衰减），标准卷积（Conv）未能很好匹配这一特性。PConv 通过非对称填充（asymmetric padding）创建水平和垂直方向的“风车叶片”状内核，从中心向外扩散，更好地捕捉小目标的中心特征，同时显著扩大感受野（receptive field），参数增加很少。

**PConv 模块结构**：
![](https://pic1.imgdb.cn/item/6a27b24fedae85a6284e7283.png)

输入张量记为 $X^{h_1 \times w_1 \times c_1}$ （$h_1, w_1, c_1$ 分别为高度、宽度、通道数）。

第一层进行**并行四路卷积**（使用分组卷积思想）：
$$X_1^{h,w,c} = \text{SiLU}(\text{BN}(W^{1 \times 3, c} X^{h_1 \times w_1 \times c_1} \ P(1\ 0\ 0\ 3)))$$
$$X_2^{h,w,c} = \text{SiLU}(\text{BN}(W^{3 \times 1, c} X^{h_1 \times w_1 \times c_1} \ P(0\ 3\ 0\ 1)))$$
$$X_3^{h,w,c} = \text{SiLU}(\text{BN}(W^{1 \times 3, c} X^{h_1 \times w_1 \times c_1} \ P(0\ 1\ 3\ 0)))$$
$$X_4^{h,w,c} = \text{SiLU}(\text{BN}(W^{3 \times 1, c} X^{h_1 \times w_1 \times c_1} \ P(3\ 0\ 1\ 0)))$$

**解释**：
- $W^{k \times l, c}$：卷积核，尺寸 $k \times l$，输出通道 $c$。
- $P(\text{left}\ \text{right}\ \text{top}\ \text{bottom})$：非对称填充参数（左右上下填充像素数），使内核呈“风车”扩散。
- `BN`（`Batch Normalization`）：批量归一化，提升训练稳定性和速度。
- `SiLU`：激活函数（`Sigmoid Linear Unit`），平滑非线性。
- 这四路分别强调不同方向（水平/垂直），匹配小目标各向异性分布。

**第一层输出特征图尺寸关系**：

$$h = \frac{h_1 + 1}{s}, \quad w = \frac{w_1 + 1}{s}, \quad c = 4c \quad (\text{或根据设计})$$

其中 $s$ 是卷积步长（`stride`）。

**拼接与最终输出**：

四路特征图拼接（`Cat`）后，通过一个 $2 \times 2$ 卷积（无填充）进一步融合：

$$X = \text{Cat}(X_1, X_2, X_3, X_4) $$
$$Y^{h_2 \times w_2 \times c_2} = \text{SiLU}(\text{BN}(W^{2 \times 2, c_2} X))$$

- 输出尺寸调整为预设 $h_2, w_2$，使其可直接替换标准 Conv 层。

- 最终输出 $Y$，兼具通道注意力效果（不同方向卷积贡献被学习）。

**参数量与感受野对比**：

标准 `Conv` 参数量（bias=False）：
$$\text{params}_{\text{Conv}} = c_2 \cdot c_1 \cdot k^2$$

对于 $3 \times 3$ 核、$c_1 = c_2 = c$：$9c^2$。

`PConv` 参数量：

$$\text{params}_{\text{PConv}} = 4 \cdot (c/4) \cdot c \cdot (3 \cdot 1) + c \cdot c \cdot (2 \cdot 2) \approx 7c^2$$

参数减少约 22.2%，感受野却大幅增加（k=3 时感受野增加 177%；实际替换 backbone 前两层时参数增加有限但收益显著）。PConv 采用分组卷积，进一步降低参数。
![|558x386](https://pic1.imgdb.cn/item/6a27b572edae85a6284e9184.png)
**效果**：PConv 增强目标-背景对比，抑制杂波（见 Fig. 4）。在底层特征提取中特别有效，适合小目标。

### 2. Scale-based Dynamic (SD) Loss（基于尺度的动态损失）

**设计动机**：小目标标注主观性强，导致 IoU（尤其是小目标）波动极大（BBox 达 86%，Mask 达 62%）。传统损失（如 CIoU、SLS）未考虑不同尺度下 Scale Loss（重叠/尺度）和 Location Loss（位置）的敏感度差异。SD Loss 根据目标大小动态调整两者权重，减少标注误差影响，提升小目标回归稳定性。

#### BBox 版本（SDB Loss）

**基础 CIoU/DIoU 形式**：
$$L_{BS} = 1 - \text{IoU} + \alpha v, \quad L_{BL} = \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2}$$

- $L_{BS}$：`Scale Loss`（基于 IoU 的尺度/重叠项）。
- $L_{BL}$：`Location Loss`（中心点距离项）。
- $\text{IoU}$：预测框与真实框交并比。
- $\rho(\mathbf{b}, \mathbf{b}^{gt})$：中心点欧氏距离。
- $c$：两框对角线长度。
- $\alpha v$：宽高比一致性惩罚。

**尺度计算**：
$$R_{OC} = \frac{w_o \cdot h_o}{w_c \cdot h_c}$$
 $w_o, h_o$：原始图像尺寸；$w_c, h_c$：当前特征图尺寸。用于还原真实目标大小。

**动态权重**：
$$\beta_B = \min(R_{OC} \cdot \delta, \delta_{\max}^{B_{gt}})$$
$$\beta_L^B = 1 - \delta \cdot \beta_B, \quad \beta_S^B = 1 + \delta \cdot \beta_B \quad (\text{或类似})$$

其中 $\delta$ 是可调超参数（控制动态范围），$B_{gt}^{\max} = 81$（小目标最大面积定义）。

**最终 SDB Loss**：

$$L_{SDB} = \beta_S \times L_{BS} + \beta_L \times L_{BL}$$

小目标（面积小）时降低 $L_{BS}$ 权重（因 `IoU` 波动大），提升 $L_{BL}$稳定性（中心点偏差小）。大目标（>81）退化为标准 `CIoU`。

#### Mask 版本（SDM Loss）

参考 SLS Loss，**Mask Scale Loss**：
$$L_{MS} = 1 - \omega(M^p, M^{gt})$$

**Mask Location Loss**（使用极坐标）：
$$
L_{ML} = 1 - \frac{\theta^p \theta^{gt}}{2 \max(d^p, d^{gt})^2 \pi} \quad (\text{或类似，强调角度和距离})
$$

- $M^p, M^{gt}$：预测/真实像素集合。
- $d, \theta$：极坐标下的距离和角度均值。
- $\omega$：像素集差异度量。

**动态权重**：

类似 BBox，但 Mask 版本**增强** \( L_{MS} \) 权重（模糊边界导致 Scale 更关键）：
$$
\beta_M = \min(R_{OC} \cdot \delta, \delta_{\max}^{M_{gt}})
$$
$$
\beta_S^M = 1 + \beta_M, \quad \beta_L^M = 1 - \beta_M
$$

**最终 SDM Loss**：

$$
L_{SDM} = \beta_S \times L_{MS} + \beta_L \times L_{ML}
$$

**注意**：Mask 版本对 batch size 敏感（因平均损失和目标大小平均），大 batch 时效果可能打折。
### 总结与优势

- **PConv**：匹配高斯分布，扩大感受野（参数高效），底层特征提取更强。
- **SD Loss**：动态平衡 `Scale` 与 `Location`，根据目标尺度自适应，缓解标注误差，尤其利于 `dim-small targets`。
- 两者可独立/联合使用，在 `YOLO`、`U-Net` 等 `backbone` 中即插即用，实验显示在 `IRSTD-1K` 和 `SIRST-UAVB` 数据集上显著提升性能。

## 三、实验结果

### 1. 数据集

1. **IRSTD-1K**：包含1,000张真实红外图像，目标尺寸较大，分辨率为512×512像素。

2. **SIRST-UAVB**：由3,000张红外图像组成，目标包括无人机和鸟类，图像采集自不同季节和天气条件下的复杂背景，具有高比例的小目标。

### 2. 结果分析
![](https://pic1.imgdb.cn/item/6a27b876edae85a6284e9366.png)
![](https://pic1.imgdb.cn/item/6a27b8afedae85a6284e9376.png)
![](https://pic1.imgdb.cn/item/6a27b8d8edae85a6284e938e.png)

- 实验结果表明，提出的PConv模块和SD Loss函数在这些数据集上均取得了显著的性能提升。PConv模块在YOLOv8n-p2检测模型和MSHNet分割模型中均表现出色，特别是在处理小目标时，能够有效提升特征提取能力和检测性能。SD Loss函数在不同尺度的目标检测中动态调整尺度和位置损失的影响系数，显著提高了网络对不同尺度目标的检测能力。

- 在SIRST-UAVB数据集上，PConv(4,3)配置提供了最佳和最平衡的性能提升，表明对于小目标，增加PConv核长度并不会带来额外的性能增益。

- 在MSHNet分割模型中，PConv显著优于其他卷积模块，表明PConv核长度为4的配置在第一层提供了更有效的感受野，对于捕获小目标特征至关重要。