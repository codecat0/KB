---
Date: 2026-11-12
tags:
  - 肝脏超声
  - 甲状腺超声
  - 乳腺超声
来源: Medical Image Analysis
年份: "2026"
机构: Macao Polytechnic University（智能医学计算实验室）、广西科技大学、中山大学、Case Western Reserve University、Eindhoven University of Technology
Code: https://github.com/PXintao/S2DENet
---

## 总结

> [!总结]
> S²DENet 通过“浅层抑制 + 深层增强”的深度差异化处理策略，结合 Multi-order Differential Convolution（MDiffConv）与 Differential Self-Attention（DiffSA），在仅 0.05M/0.15M 参数下实现十个公开超声数据集上接近或超越 SOTA 的分割性能，并保持 80+ FPS 实时推理。

### 摘要

超声图像分割是临床诊断的基石，但受斑点噪声与边界模糊等固有伪影影响，仍是极具挑战的任务。现有方法通常在所有网络层采用统一的特征提取策略，忽视了噪声主导的浅层与语义丰富的深层之间的本质差异。这种单一策略迫使网络同时学习噪声抑制与特征增强，常导致过参数化，并在资源受限环境中严重牺牲计算效率。为解决这些问题，本文提出 S²DENet——一种在浅层进行噪声抑制、在深层进行语义特征增强的高效网络架构。具体而言，我们设计了多阶差分卷积（Multi-order Differential Convolution, MDiffConv）以增强高频特征捕获能力，并在浅层实施抑制策略、在深层实施增强策略；同时引入差分自注意力（Differential Self-Attention, DiffSA）机制以抑制噪声伪影并保持结构完整性，深层则切换为标准自注意力以放大语义特征。这种深度差异化设计结合差分机制，使模型能在不同网络阶段专注于特定任务，从而降低学习复杂度并提升特征表征效率。在十个公开超声数据集上的实验结果表明，S²DENet 在效率与精度之间取得了卓越平衡，在九个公开数据集上达到 SOTA 性能。仅以 0.05M/0.15M 参数（相较 SOTA 方法减少超过 99%），S²DENet 在 NVIDIA RTX A8000 GPU 上保持 80+ FPS 实时推理，同时取得具有竞争力或更优的性能。S²DENet 不仅为超声图像分割建立了新范式，也为临床实践部署开辟了新可能。代码见 https://github.com/PXintao/S2DENet。

---

## 要解决的问题与研究动机

### 核心问题：

超声图像分割长期受斑点噪声（speckle noise）与边界模糊困扰。现有 CNN、Transformer 及混合架构普遍采用**统一特征提取策略**（所有层用同一套操作），忽略了浅层与深层特征本质差异：
- **浅层**：噪声主导，高分辨率，高频成分中噪声与真实边界混杂。
- **深层**：语义丰富，噪声经多层滤波后大幅衰减。

这种“一刀切”策略迫使网络同时学习噪声抑制与语义增强，导致过参数化、效率低下，难以在资源受限临床环境部署。

### 研究动机：

作者用 ResNet18、DenseNet121、MobileNetV2 做了对照实验（Table 1）：
![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794454694659.webp]]
- 浅层→深层，MSE 显著下降（噪声被抑制）。
- SSIM 明显上升（结构信息更清晰）。

结论：浅层核心任务是“对抗噪声”，深层核心任务是“提取语义”。因此需要**深度差异化处理**而非统一范式。

频率域视角：斑点噪声与高分辨率边界细节交织在高频，低频携带粗略解剖结构。统一策略难以兼顾，往往靠堆参数补偿。

---

## 方法拆解

### 整体架构（Fig. 3）
![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794454739497.webp]]

S²DENet 采用四阶段编码器 + 双路径解码器：
- **浅层（L1, L2）**：抑制策略（噪声压制 + 结构保护）
- **深层（L3, L4）**：增强策略（边界细化 + 语义放大）
- 初始 5×5 卷积扩展通道至 base dim *c*（S²DENet: *c*=8；Tiny: *c*=4）
- 频率解耦：固定 3×3 Laplacian 高通 + 3×3 平均低通滤波
- 双路径解码：独立边界解码器 + 掩码解码器，逐层融合


### 核心模块 1：Multi-order Differential Convolution（MDiffConv）

**设计思路**：高频通路中噪声与真实边界混杂，不能硬过滤。采用“结构门控的自适应高频调制”。MDiffConv 由二阶差分卷积后接多个一阶差分卷积串联构成（Fig. 4），利用差分算子对梯度变化的敏感性来捕获高频信息，同时通过可学习权重实现软门控，避免对真实弱边界的硬抑制。
![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794474610652.webp]]

**二阶差分卷积的数学推导（以 3×3 局部区域为例）**：

对于中心像素 \(x_5\) 及其对称位置像素，二阶差分形式为 \(x_1 + x_3 - 2x_5\)。将差分操作与卷积结合，可写成：

$$
\begin{bmatrix}
x_1+x_3-2x_5 & x_2+x_6-2x_5 & x_3+x_9-2x_5 \\
x_4+x_2-2x_5 & 0 & x_6+x_8-2x_5 \\
x_7+x_1-2x_5 & x_8+x_4-2x_5 & x_9+x_7-2x_5
\end{bmatrix} * W \tag{1}
$$
展开卷积运算得：
$$
(x_1+x_3-2x_5)w_1 + (x_2+x_6-2x_5)w_2 + \cdots + (x_9+x_7-2x_5)w_9 \tag{2}
$$
利用分配律并按像素位置重新分组后：
$$
\begin{align}
& x_1(w_1+w_7) + x_2(w_2+w_4) + x_3(w_3+w_1) + x_4(w_4+w_8) \nonumber\\
& + x_5\Bigl(-2\sum_{i\neq5}w_i\Bigr) + x_6(w_6+w_2) + x_7(w_7+w_9) \nonumber\\
& + x_8(w_8+w_6) + x_9(w_9+w_5) \tag{3}
\end{align}
$$

这表明差分卷积在数学上等价于一个标准卷积，其等效核 $W^*$ 为：
$$
\begin{bmatrix}
x_1 & x_2 & x_3 \\
x_4 & x_5 & x_6 \\
x_7 & x_8 & x_9
\end{bmatrix}
*
\begin{bmatrix}
w_1+w_7 & w_2+w_4 & w_3+w_1 \\
w_4+w_8 & -2\sum_{i\neq5}w_i & w_6+w_2 \\
w_7+w_9 & w_8+w_6 & w_9+w_5
\end{bmatrix}
\tag{4}
$$

因此，训练与推理阶段均可直接使用标准卷积实现，无需自定义算子，计算开销与普通卷积一致。

- **浅层**：抑制斑点伪纹理，保留真实解剖轮廓
- **深层**：增强边界细节与关键解剖结构

后续通过 Multi-scale ConvPool（Fig. 5）融合边界线索与平滑结构特征（多尺度卷积 + 大感受野平均池化），再与低频通路逐元素相乘，实现边界与结构的稳健整合。
![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794474664448.webp]]
### 核心模块 2：Differential Self-Attention（DiffSA）

![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794476374912.webp]]

**问题**：标准自注意力在噪声环境下易对高幅值斑点产生错误响应，导致注意力权重偏向噪声区域。

**解决**：对 Query 与 Key 本身先做差分，消除共同模式噪声后再计算注意力。公式如下：

$$
\text{DiffSA}(Q,K,V) = \text{softmax}\left(\frac{(Q_1 - \lambda Q_2)(K_1 - \beta K_2)^\top}{\sqrt{d_k}}\right) V \tag{5}
$$
其中：
- $Q_1,Q_2$ 与 $K_1,K_2$ 由同一输入特征图经相似卷积生成，必然共享来自斑点噪声的共同分量；
- $\lambda=\sigma(\tilde{\lambda})$、$\beta=\sigma(\tilde{\beta})$ 为可学习标量（初始化为 1.0，经 sigmoid 约束到 (0,1)），自适应决定从主表示中减去次级表示的比例；
- $d_k$ 为 Key 向量维度，除以 $\sqrt{d_k}$ 起梯度稳定作用；
- 仅需一次矩阵乘法，计算高效，契合轻量化设计。

- **浅层**：使用 DiffSA 保护结构完整性，减少错误注意力响应
- **深层**：切换为标准 Self-Attention，充分挖掘已去噪后的丰富语义信息

每个 DiffSA 模块配合标准残差连接与层归一化，并与前馈网络构成完整处理单元。

### 解码器（Fig. 7）
![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794476411961.webp]]
- 独立边界解码器与掩码解码器
- 逐尺度特征交互 + 自适应加权融合 + 归一化
- 边界约束直接注入掩码路径，提升边界精度与一致性

---

## 训练与推理流程

### 训练设置
- **优化器**：AdamW（lr=0.001, β1=0.9, β2=0.999, weight decay=0.01）
- **学习率调度**：Cosine Annealing，最大 300 epoch，最小 lr=0.0003
- **Batch size**：8
- **损失**：BCE + Dice（权重均为 1）
- **数据增强**：旋转、翻转、亮度/对比度调整、噪声等
- **输入分辨率**：统一 resize 到 256×256
- **硬件**：NVIDIA RTX A8000 GPU，PyTorch

### 推理
- 固定阈值 0.5 二值化
- 实时性能：S²DENet ≈ 81 FPS；Tiny 版本更高
- 内存占用极低（约 10 MB）

Foundation Models（MedSAM、SAM-Med2D）使用官方预训练权重 + GT 衍生提示（带随机抖动）进行零样本推理评估。

---

## 实验设置与关键结果

### 数据集（Table 2）
覆盖乳腺、肾脏、胎儿头部、颈动脉、甲状腺、肌肉、心脏等 12 个公开数据集（主实验用 10 个有完整划分的）：

| 数据集 | 解剖部位 | 总数 | 备注 |
|--------|----------|------|------|
| BUSI / BUSIS / BUS_BRA / UDIAT | 乳腺 | 780 / 562 / 1425 / 163 | 多中心 |
| KidneyUS | 肾脏 | 534 | - |
| Fetal-HC | 胎儿头围 | 999 | - |
| CCAU | 颈动脉 | 1020 | - |
| DDTI / TN3K | 甲状腺 | 466 / 3493 | - |
| FALLMUD | 肌肉 | 813 | - |
| EchoNet-ED / CAMUS | 心脏 | 10024 / 500 | 大规模 |

### 主实验结果（Table 3 & 4）
![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794476467861.webp]]
![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794476501453.webp]]

- **S²DENet** 在 7 个数据集上取得最高 Dice，平均 AvgDice **88.68%**、Avg mIoU **80.24%**
- 参数量仅 **0.15M**（Tiny 0.05M），MACs 0.77G / 0.27G
- 相比 U-Net（AvgDice 74.59%）提升约 14 个百分点；相比 MobileUViT（87.65%）仍领先
- 参数量较多数 SOTA 减少 **>99%**，同时保持或超越精度
- 实时性：81 FPS（RTX A8000）

### 消融实验（Table 5）
| Layered | DiffSA | MDiffConv | Dice | 增益 |
|---------|--------|-----------|------|------|
| - | - | - | 76.06 | baseline |
| ✓ | - | - | 77.42 | +1.36 |
| - | ✓ | - | 78.23 | +2.17 |
| - | - | ✓ | 78.68 | +2.62 |
| ✓ | ✓ | ✓ | **80.77** | **+4.71** |

完整组合产生真正的协同增益。

### 浅层抑制层选择（Table 6）
![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794476534143.webp]]

仅对 L1+L2 做抑制效果最佳（80.77 Dice）；扩展到更深会下降，验证“浅抑深增”必要性。

### 跨数据集泛化（Table 7）
![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794476555126.webp]]

- BUSI→BUSIS：85.63 Dice（最优）
- DDTI→TN3K：52.30（次优，差距 2.76）
- EchoNet-ED→CAMUS/EchoNet：整体竞争力强，参数仅 0.15M 却接近大模型表现

### 其他验证
![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794476720309.webp]]

- 噪声鲁棒性（Fig. 10、Appendix Table A.1）：DiffSA 与 MDiffConv 在 Gaussian、Impulse、K-Distribution（模拟斑点）噪声下均显著优于标准对应模块![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794476720309.webp]]
- 深层增强可视化（Fig. 11）：标准 SA 与增强版 MDiffConv 在深层产生更集中的语义激活![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794476748888.webp]]
- 跨模态初步结果（Appendix Table A.3）：在 ACDC（MRI）与 KiPA2022（CT）上无需调参即表现良好![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794476813969.webp]]

---

## 亮点、局限、适用边界

### 亮点
1. **范式创新**：首次系统提出并验证“浅层抑制 + 深层增强”的深度差异化处理，打破统一特征提取惯性。
2. **极致轻量**：0.05M/0.15M 参数实现 SOTA 级精度与 80+ FPS，真正适合床旁/便携超声。
3. **模块可解释且高效**：差分操作可重参数化为标准卷积；DiffSA 仅增加极少计算。
4. **全面验证**：10 个公开数据集 + 跨数据集 + 噪声鲁棒 + 初步跨模态。
5. **代码开源**。

### 局限
1. **回顾性评估为主**：尚未进行前瞻性多中心临床验证。
2. **极端低对比度失败案例**（Fig. 12）：抑制可能局部过强，导致边界欠分割。![[S²DENet Shallow suppression and deep enhancement network for general ultrasound image segmentation-1794476774414.webp]]
3. **跨模态结果 preliminary**：仅在有限设置下测试 MRI/CT，系统性验证不足。
4. **固定浅/深划分**：L1-L2 抑制、L3-L4 增强是经验最优，但非自适应。
5. **与医学基础模型结合**：需架构级改造（非简单插件），作者将其列为未来工作。

### 适用边界
- **强适用**：资源受限临床环境、实时床旁超声、需要高边界精度的浅表器官（乳腺、甲状腺、颈动脉等）、对噪声敏感的场景。
- **需谨慎**：极端低对比度/重度阴影图像；需要超强跨域零样本能力时，大基础模型仍可能更优（但计算代价高）。
- **潜在扩展**：其他噪声主导的医学模态（部分 OCT、内窥镜等）。

---

## 对实际应用或研究延展的判断

### 实际应用潜力
S²DENet 是目前超声分割领域**精度-效率帕累托前沿**上的突出工作。0.15M 参数 + 81 FPS 使其极易嵌入便携超声设备、移动端 AI 辅助诊断、术中实时引导系统。对基层医疗机构与资源有限场景尤其有价值。

### 研究延展方向
1. **自适应深度策略**：根据输入图像噪声水平或内容自动决定抑制/增强层数。
2. **测试时校准**：针对设备/协议引起的强度分布偏移，设计轻量归一化对齐或测试时适应。
3. **与基础模型融合**：将深度差异化思想注入 SAM 类医学基础模型的图像编码器与掩码解码器。
4. **多任务/多模态扩展**：边界 + 分割联合、超声视频序列、与其他模态联合训练。
5. **前瞻性临床研究**：多中心、多设备、专家评估的真实部署验证。

整体判断：这是一篇**方法论扎实、实验全面、工程落地性极强**的工作，有望成为轻量级医学图像分割的重要参考范式。

---

## 术语解释

| 术语 | 解释 |
|------|------|
| **Speckle noise（斑点噪声）** | 超声成像中由相干波干涉产生的乘性噪声，表现为颗粒状纹理，严重干扰边界与组织表征。 |
| **MDiffConv（Multi-order Differential Convolution）** | 多阶差分卷积。通过重参数化将差分算子嵌入标准卷积核，浅层抑制噪声、深层增强边界。 |
| **DiffSA（Differential Self-Attention）** | 差分自注意力。对 Q/K 做差分以消除共同模式噪声，再计算注意力，提升噪声鲁棒性。 |
| **Depth-differentiated / Hierarchical processing** | 深度差异化/层次化处理。浅层与深层采用不同策略（抑制 vs 增强），而非统一操作。 |
| **Frequency decoupling** | 频率解耦。用高通/低通滤波将特征分为高频（边界+噪声）与低频（结构）通路分别处理。 |
| **Re-parameterization** | 重参数化。训练时使用复杂结构，推理时等价转换为标准卷积，保持效率。 |
| **AvgDice / Avg mIoU** | 多个数据集上的平均 Dice 系数与平均交并比，衡量整体性能。 |
| **Foundation Models（MedSAM / SAM-Med2D）** | 基于 SAM 的医学领域大模型，零样本/提示引导分割，参数量大。 |



