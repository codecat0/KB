---
Date: 2026-06-09
tags:
  - 分割
  - 脑卒中
  - 脑肿瘤
  - 胰腺癌
来源: MIA
年份: "2025"
机构: 中南大学、东南大学、华中科技大学、南京医科大学第一附属医院、南京医科大学
Code: https://github.com/hulinkuang/LW-CTrans
---
## 摘要：

**背景与动机**

- **现状：** 近期基于卷积神经网络（CNN）和Transformer的模型在3D医学图像分割领域取得了很有希望的性能。
- **问题：** 然而，即使这些模型参数量很大，它们在**分割微小目标（small targets/lesions）** 方面表现不佳。
- **目标：** 因此，作者设计了一种新颖的**轻量级混合网络**，名为LW-CTrans，它结合了CNN和Transformer的优势，并且能够在不同阶段提升网络的全局（Transformer的长程依赖捕获能力）和局部（CNN的细节捕获能力）表征能力。

**核心模块与架构**

`LW-CTrans`主要包括一个混合编码器（Hybrid Encoder）和一个轻量级解码器（Decoder），其关键设计如下：

- **动态Stem (Dynamic Stem)：** 首先设计了一个动态Stem模块。这个模块可以**适应不同分辨率的输入图像**，提高了模型的通用性和鲁棒性。
- **混合编码器的第一阶段 (First Stage of Encoder)：**
    - 目标：以更少的参数捕获局部特征。
    - 提出了**多路径卷积块（MPConv block）**。这个模块在编码器初期阶段主要用于捕获局部特征，同时减少参数量。
- **混合编码器的中间阶段 (Middle Stages of Encoder)：**
    - 目标：同时学习全局和局部特征。
    - 提出了基于**多视角池化（Multi-View Pooling based Transformer，MVPFormer）的Transformer模块。MVPFormer通过将3D特征图投影到三个2D子空间**来处理小目标问题。同时，这些中间阶段也使用了**MPConv块**来增强局部表征学习。
- **混合编码器的最终阶段 (Final Stage of Encoder)：**
    - 目标：主要捕获全局特征。
    - 仅使用提出的**MVPFormer**模块。
- **解码器 (Decoder)：**
    - 目标：减少解码器的参数量。
    - 提出了**多阶段特征融合模块（multi-stage feature fusion module）**

**实验结果与贡献**

作者在三个公共数据集上对LW-CTrans进行了广泛实验，涵盖了三个任务：卒中病变分割（stroke lesion segmentation）、胰腺癌分割（pancreas cancer segmentation）和脑肿瘤分割（brain tumor segmentation）。

- **分割性能：**
    - LW-CTrans在这三个数据集上分别取得了62.35±19.51%、64.69±20.58%和83.75±15.77%的Dice分数。
    - 性能**优于16种先进（state-of-the-art）方法**。
- **轻量化特性：**
    - 在三个数据集上的参数量分别为2.08M、2.14M和2.21M。
    - 这些参数量**小于非轻量级3D方法**，并且**接近轻量级方法**的参数量。
- **小目标分割：**
    - LW-CTrans在**小病变分割**方面也取得了最佳性能。

## 1. 引言

**3D医学图像分割的背景和重要性**

- **目标与作用：** 3D医学图像分割旨在从计算机断层扫描（CT）或磁共振成像（MRI）等扫描图像中精确地描绘出器官、组织和病灶的区域。这在许多疾病（如中风和脑肿瘤）的诊断和预后中起着关键作用。
- **现有挑战与需求：** 然而，手动分割3D医学图像既繁琐又耗时。因此，临床实践迫切需要全自动化且有效的3D医学分割方法。

**深度学习方法的发展：** 近十年来，最流行的医学图像分割方法主要基于深度学习技术，包括：

- **不同类型的分割：** 弱监督分割、全监督分割、领域适应、半自动分割和全自动分割。
- **CNN方法的优势与局限：**
    - 卷积神经网络（CNN）方法（例如AttnUNet3D和nnUNet）常用于全自动医学图像分割。
    - **局限性：** 卷积操作固有的归纳偏置（inductive bias）使其难以建模长距离依赖关系（即全局特征）。
- **Transformer方法的引入：**
    - Transformer因其建模全局特征的能力而广受关注，并被广泛应用于医学图像分割，例如Dformer。

**混合（Hybrid）CNN-Transformer方法的研究现状与挑战：** 为了结合CNN的局部特征捕捉能力和Transformer的全局特征建模能力，许多研究人员致力于探索如何整合两者以实现更好的医学图像分割。

- **常见的混合方法：**
    - **级联方法（Cascading Approach）：** 大多数现有混合模型习惯采用级联方法。例如，TransUNet、CoTr和BATFormer首先使用CNN编码器进行特征提取，然后将这些特征输入Transformer编码器来建模长距离依赖关系。
    - **问题：** 这种方法缺乏局部特征和全局特征之间的有效融合。
- **并行方法（Parallel Approach）：** 其他工作如TransHRNet、PHTrans和FAT-Net使用并行的CNN和Transformer编码器分别提取局部和全局特征。
- **非对称方法：** UNETR则使用Transformer作为编码器，CNN作为解码器。
- **现有混合模型的挑战：** 尽管这些方法取得了可喜的分割性能，但大多数没有探索如何设计一个**统一的混合编码器**，用于在编码器的每个阶段学习不同的有效特征（局部和/或全局特征）。
- **结论：** 最佳混合网络的设计仍然是一个开放的挑战。

**轻量化网络的需求与现状**

- **临床需求：** 在临床实践中，理想的医学图像分割方法应该是**轻量级的**，同时其分割性能要接近现有最先进水平（SOTA）。
- **轻量化研究：** 一些研究人员探索了如何为3D医学图像分割设计轻量级模型。
- **现有轻量级模型：**
    - **基于CNN的：** 多数轻量级方法是基于CNN实现的，例如LCOVNet和ADHDC-Net。
    - **基于混合架构的：** 基于混合CNN和Transformer的轻量级方法较少，例如SlimUNETR。
    - **性能对比：** 现有轻量级模型（参数少于3M）在某些任务（如中风病灶分割）上的性能不如非轻量级的SOTA模型（如nnUNet）
- **本文的研究重点：** 专注于设计一种**混合方法**，该方法在保持**轻量化**的同时，实现**更好的分割性能**。

**小病灶分割的挑战与目标**

- **现实场景中的挑战：** 在中风和脑肿瘤等实际临床场景中，小病灶难以准确分割。
- **重要性：** 准确分割小病灶在临床实践中至关重要。例如，精确地定位小中风病灶可以帮助医生指导机械取栓。
- **现有方法的不足：** 大多数当前分割方法难以描绘这些小病灶。
- **本文的目标：** 也将开发一种**针对小目标的定制方法**，并将其合理地整合到所设计的轻量级分割模型中。

**LW-CTrans模型的提出**

本文提出了一个新颖的**轻量级CNN和Transformer混合网络（LW-CTrans）**，旨在以更少的参数实现足够好的3D医学图像分割性能，并具备良好的小目标分割能力。

- **关键创新点：**
    - 设计了**轻量级多路径卷积块（MPConv）** 来捕获局部信息。
    - 提出了基于**多视图池化（MVPFormer）** 的新型Transformer块，它轻量化且能提升小目标分割性能。
    - 在基于CNN的解码器中，设计了**多阶段特征融合模块**以进一步减少参数。
- **实验结果：** 实验证明，LW-CTrans在三个3D医学分割任务上，性能优于或可媲美15个SOTA方法，同时参数数量极少，并且能很好地分割小病灶。

## 2. 相关工作

### 2.1. 基于 CNN 的分割方法

在 Transformer 出现之前，CNN 多年来一直是图像分割的首选模型。CNN 方法的特点在于擅长捕获**局部信息**，但也因此不擅长捕获**全局信息**。

- **UNet++ :** 设计了一系列**嵌套和密集跳跃连接路径** (nested and dense skip pathways)，以减少编码器和解码器之间的语义差距。
- **Attention U-Net :** 提出了**注意力门** (attention gate)，使模型能够更专注于不同尺寸和形状的目标区域。
- **CE-Net :** 提出了一个上下文编码器网络 (context encoder network)，包含一个**密集空洞卷积块** (dense atrous convolution block) 和一个**残差多核池化块** (residual multi-kernel pooling block) 以获取更多高层信息。
- **nnUNet :** 设计了一个 3D 医学图像分割框架，能够**自动确定网络架构和超参数**。
- **Double U-Net :** 采用了**空洞空间金字塔池化** (atrous spatial pyramid pooling) 来捕获多尺度信息。

为了弥补 CNN 在全局信息捕获上的不足，一些方法进行了改进：

- **RepLKNet :** 利用了**大核卷积** (large kernel convolution)。
- **X-net :** 设计了**非局部操作** (nonlocal operation)，特别是特征相似性模块 (feature similarity module)，来捕获**长距离依赖** (long-range dependencies)。

### 2.2. 基于 Transformer 的分割方法

自从 **Vision Transformer (ViT)** 成功应用于计算机视觉任务以来，越来越多的研究人员开始探索 Transformer 在该领域的应用。Transformer 以其强大的**全局特征建模**能力而闻名。

- **Segformer:** 设计了一个**分层的 Transformer 编码器**，能够输出适用于密集预测任务的**多尺度特征**。
- **SwinUNet :** 设计了一个**纯粹基于 Transformer** 的 Swin Transformer，用于医学图像分割网络。
- **MP-TransUNet:** 在 SwinUNet 的基础上提出了一个**多尺度特征编码器**，并利用 Transformer 设计了一个**多尺度特征融合模块**。
- **Dual-ViT:** 引入了一个关键的语义路径 (critical semantic pathway)，可以更有效地将 Token 向量压缩成**全局语义**，同时降低计算复杂度。
- **Mobile-ViT:** 设计了一个**轻量级且通用**的视觉 Transformer，适用于移动设备。

**局限性：** 基于 Transformer 的方法通常需要**大量的训练数据**才能实现卓越的分割性能。

### 2.3. 混合模型用于分割

近期的研究表明，结合卷积神经网络（CNN）和Transformer可以吸收这两种架构的优点，从而在医学图像分割任务中取得更好的性能。

- **TransUNet ：** 首次将Transformer引入CNN中，用于建模**长距离依赖关系**（long-range dependencies）。
- **CoTr：** 为了降低计算复杂性，在编码器中设计了一个**3D可变形Transformer**（3D deformable transformer）。
- **UNETR：** 利用Transformer作为**编码器**来学习长距离依赖关系，并使用CNN作为**解码器**。
- **nnFormer：** 使用Transformer作为**主干网络**（backbone），并采用CNN进行**下采样**。
- **TransHRNet ：** 设计了一种更有效的Transformer块，即使在**高特征分辨率**下也能有足够的特征表示能力。
- **PHTrans：** 在深层阶段使用**并行混合模块**，独立地学习**局部信息**和**全局信息**。
- **HiFormer：** 设计了一种新颖的方法，用于高效地连接CNN和Transformer，以进行医学图像分割。
- **FAT-Net：** 设计了**并行**的CNN和Transformer编码器，用于捕获长距离依赖关系和全局上下文信息。
- **AST-Net：** 在编码器中同时使用了**轻量级卷积模块**和**轴向空间Transformer模块**，以降低计算复杂性。

**局限性：** 现有的基于混合模型的分割方法通常具有**大量的参数**，这不利于它们在临床实践中的应用。

### 2.4. 轻量级分割网络

**核心目标：** 轻量级网络旨在最大限度地减少模型参数并提高深度网络的效率。

**轻量化CNN的研究：**

- **卷积分解 (Convolution factorization)：** 是一种减少卷积计算负荷的直观方法。
- **MobileNet ：** 使用**深度可分离卷积**（deep convolution）和**点式卷积**（point-wise convolution）来构建轻量级模型，使其可应用于移动设备。
- **ShuffleNet：** 进一步采用了**分组点式卷积**（grouped point-wise convolution）和**通道混合**（channel-mixing）来降低计算成本。
- **LCOVNet：** 引入了一个**轻量级基于注意力的卷积块**，该块包含一个**时空可分离卷积分支**来减少参数，以及一个**轻量级特征校准分支**来增强学习能力。
- **ADHDC-Net ：** 提出了一种带有**注意力机制**的轻量级自动3D算法，用于医学图像分割。

**轻量化Transformer的研究：**

- **Mobile-Former ：** 使用**有限数量的token**（例如6个或更少，随机初始化）来捕获全局先验知识，从而减少计算开销。
- **BATFormer：** 引入了**跨尺度全局Transformer模块**，同时利用多个小尺度特征图，在增强全局特征的同时降低了计算复杂性。
- **SlimUNETR ：** 设计了一个新的Transformer块，通过**自注意力机制分解**来降低计算复杂性。

**研究空白：** 目前，关于基于CNN和Transformer混合架构的轻量级模型，以提高分割性能的研究还很少。

## 3. 方法

### 3.1. 概述
![](https://pic1.imgdb.cn/item/6a27ce7dedae85a6284f1ad0.png)
- **LW-CTrans 架构**：所提出的 LW-CTrans 是一种常用的 UNet 变体，包含一个混合编码器（hybrid encoder）和一个解码器（decoder），其架构在图 2 中有所展示。
- **混合编码器结构**：该编码器包含一个**动态 Stem 模块**，它能够适应不同尺寸的输入图像。随后是四个阶段用于捕获不同尺度的特征：
    - 一个**多路径卷积（multi-path convolution）阶段**。
    - 两个**混合阶段（hybrid stages）**（结合了 CNN 和 Transformer）。
    - 一个 **MVPFormer 阶段**（Multi-View Pooling based Transformer）。
- **解码器结构**：解码器包含一个**多阶段特征融合（MSFF）模块**，以及与编码器 Stem 阶段对称的**解码器层（decoder levels）**。
- **目标**：为了获得针对不同形状和大小目标的更好特征表示

### 3.2. 动态 Stem (Dynamic Stem)

- **背景问题**：Stem 架构通常用于提升模型性能，但大多数现有方法使用固定的两到三层卷积。然而，不同数据集的图像分辨率各异，对于低分辨率数据集，过多的卷积层会导致在进入后续阶段之前过度下采样，从而可能导致性能下降。
- **解决方案**：LW-CTrans 遵循 nnUNet 的设计，通过分析数据集的属性，设计了一个**动态 Stem 架构**来确定 Stem 模块中卷积层的数量。
- **确定卷积层数量的步骤**：
    1. 将 Patch Size 初始化为重新采样图像的中位数形状。
    2. 根据图像形状和体素间距，确定每个轴上的下采样步数。
    3. 如果分辨率变化显著，独立地对每个轴进行下采样，直到分辨率匹配。

### 3.3. 多路径卷积块 (Multi-path Convolution Block)

- **早期阶段的需求**：在网络的早期阶段，模型应该更侧重于局部细节，例如目标的边界，这意味着基于 CNN 的模块是最佳选择。
    
- **Stem 阶段**：在 Stem 阶段，特征图的通道数较少，标准卷积更适合捕获图像的局部特征。
    
- **设计轻量级卷积**：随着特征图通道数的逐渐增加，为了保持模型的轻量化，作者设计了一种新颖的轻量级卷积块。**替代 Depth-wise Separable Convolution (DSC)**：传统的 DSC (如 MobileNet 中使用) 有助于模型轻量化，但可能导致信息丢失，尤其是在复杂任务中，并且可能损害表示能力和准确性。
    
- **MPConv 的优势**：作者提出了**多路径卷积（MPConv）块**来取代 DSC，以提高模型性能。MPConv 沿着不同的方向使用**深度卷积 (DWConv)** 来从多个角度学习特征，从而提取出更稳健和信息更丰富的表示。
    
- **MPConv 结构**：
    ![|522|522x414](https://pic1.imgdb.cn/item/6a27cfcdedae85a6284f1c04.png)
    1. 一个 DWConv 层。
    2. 随后是三个具有不同核大小的 DWConv 层。
    3. 一个**逐点卷积（PWConv）** 层。
    4. **标准化和激活**：每个 DW 卷积后面都跟着一个 Instance Normalization（实例归一化）和一个 LeakyReLU 激活函数。


- **数学公式**：描述了 `MPConv 块`如何计算输入特征图 $X$ 的输出 $X_{out}$：
    - 首先，输入 X 经过一个 `3x3x3` 的 `DWConv`，然后是 `Instance Normalization` ($B$) 和 LeakyReLU ($δ$)，得到中间特征 $\hat X$。
        
        $$ \hat X=δ(B(C_{DW}^{3×3×3}(X))) $$
        
    - 然后，$\hat X$ 通过三个不同核大小的 `DWConv`（分别对应 $i=1,2,3$ 的 $C_{DW}^i$），捕获轴向（$D$）、冠状（$H$）和矢状（$W$）视图的特征 $X_D,X_H,X_W$。
        
        $$ X_D,X_H,X_W=δ(B(C_{DW}^i(X~))),i=1,2,3 $$
        
        - $C_{DW}^i$ 代表的核大小分别为 $1×3×3$（轴向特征）、$3×1×3（$冠状特征）和 $3×3×1$（矢状特征）。
    - 最后，将这三个视图的特征进行拼接 (`CONCAT`)，并通过一个 `PWConv` ($CPW$) 得到最终输出 $X_{out}$。
        
        $$ X_{out}=C_{PW}([X_D,X_H,X_W]) $$
        

### 3.4. 多视角池化 Transformer (MVPFormer) 模块

**背景与动机：** Transformer模型在捕获**全局信息**方面表现出色，但其高昂的计算复杂度限制了它在3D医学图像分割中的应用。

- 最近的研究表明，一些参数量较少的操作，如大核卷积、特征平移和**池化操作**，可以替代多头自注意力机制（Multi-Head Self-Attention），同时仍能达到相似的良好性能。
- 此外，对于**小目标**（small targets）的分割，由于正负体素之间存在高度不平衡（high imbalance between positive and negative voxels），使得准确分割小目标非常困难。将3D空间投影到2D子空间可以在一定程度上缓解小目标的数据不平衡问题。

**MVPFormer的设计理念：**

- 受医生通常从三个正交视图（轴位、矢状位和冠状位）解释图像的启发，作者认为将特征图投影到这三个视图可以最大限度地保留有用的空间特征，同时过滤掉其他不重要的特征。
    
- 因此，作者设计了一个**轻量级**的**多视角池化 Transformer (MVPFormer)** 模块，它基于池化操作和三个医学图像视图，用于捕获有效的全局信息、处理小目标，并减少参数量。
    
- **多视角池化 (MVP) 操作：** MVP操作用于替代自注意力机制。
    ![](https://pic1.imgdb.cn/item/6a27d02dedae85a6284f1c4b.png)
    1. **投影到2D子空间：** 首先，通过**全局平均池化 (GAP)** 操作，将3D空间投影到三个2D子空间：轴位（Axial）、矢状位（Sagittal）和冠状位（Coronal）视图。
    2. **特征组合：** 随后，使用**拼接 (concatenation)** 操作和**另一次GAP**操作来组合这三个视图的特征。


`MVP`操作的数学公式如下：

$$ X_S=SagittalGAP(X)\\X_A=AxialGAP(X)\\X_C=CoronalGAP(X)\\MVP(X)=GAP(CONCAT(X_S,X_A,X_C)) $$

- 其中，$X$ 是输入特征图，$SagittalGAP$、$AxialGAP$ 和 $CoronalGAP$ 分别表示沿矢状面、轴向和冠状面的`GAP`操作，`CONCAT` 表示沿**通道轴**的拼接。

**MVPFormer 模块的输出：**

在用提出的MVP操作替换自注意力后，第 $l$ 个MVPFormer块的输出计算如下 ：

$$ \hat x^l=MVP(x^{l−1})+x^{l−1}\\x^l=MLP(\hat x^l)+\hat x^l

$$

其中 $x^{l−1}$ 是来自第 $l-1$ 块的输入，$\hat x^l$ 是MVP的输出，$x^l$ 是第 $l$ 个MVPFormer块的最终输出。

### 3.5. CNN 和 Transformer 的混合模块

**背景与动机：** 在混合编码器的中间阶段，网络学习到的特征应该逐渐从**局部细节特征**转向**全局高级特征**。因此，作者在中间阶段混合使用了**MPConv**（多路径卷积块，用于局部特征）和 **MVPFormer**（用于全局特征），以同时提取局部和全局特征。

**混合模块的实现：**

- 采用**通道分离策略**（Channel Separation Strategy），将输入特征分别送入MPConv和MVPFormer。然后，将这两个模块的输出融合，作为混合模块的输出。
- 为了进一步减少参数量，在这个混合模块中执行了**通道分割**（channel split）：一半的通道作为MVPFormer的输入，剩下的一半作为MPConv的输入。
- 由于串联（Sequential）的CNN和Transformer混合模型主要学习全局或局部信息，作者采用了**并行**的MPConv和MVPFormer块，然后进行拼接（concatenation）。

**并行混合模块的数学公式如下：**

$$ X_T,X_C=ChannelSplit(X)\\\hat X_T=MVPFormer(X_T)\\\hat X_C=MPConv(X_C)\\X_{out}=CONCAT(\hat X_C,\hat X_T) $$

- 其中，$X_T$ 和 $X_C$ 分别表示MVPFormer和MPConv块的输入，$\hat X_T$ 和 $\hat X_C$ 分别表示它们的输出。

### 3.6. 解码器设计 (Decoder design)

为了平衡模型的复杂性（使其更轻量化）和分割性能，作者设计了一个包含**多阶段特征融合 (Multi-stage feature fusion, MSFF)** 模块的解码器，并且其解码器层与编码器中的 **Stem 模块** 对称。

为了使解码器更加轻量化，作者没有采用常见的逐步（step-by-step）上采样编码器特征来获取分割结果。相反，他们提出了一种策略：编码器不同阶段学习到的特征包含丰富的多尺度信息，通过融合这些不同尺度的特征来获得分割结果。

**具体实现如下：**

- 将编码器中 **Stage 2 到 Stage 4** 学习到的特征 $X_2,X_3,X_4$ 上采样到 **Stage 1** 的大小。
    
- 上采样过程是通过 **点卷积 (Point-wise Convolution, PWConv)** 和 **三线性插值 (trilinear interpolation)** 函数 $f$ 实现的。
    
- 然后，将这些上采样后的特征与 Stage 1 的特征 $X_1$（经过一次PWConv处理）**进行拼接 (CONCAT)**。
    
- 最后，拼接后的特征再通过一次PWConv进行处理，计算出融合特征 $Y$。
    
    $$ Y=C_{PW}(CONCAT(f(X_2),f(X_3),f(X_4),C_{PW}(X_1))) $$
    
    - 其中，$C_{PW}$ 表示点卷积，$f$ 表示点卷积后接三线性插值函数，$X_i$ 表示第 $i$ 阶段的特征。

### 3.7. 损失函数 (Loss function)

采用了 **Dice 损失 (dice loss)** 和 **交叉熵损失 (cross-entropy loss)** 的组合作为最终的损失函数。

- **Dice 损失:**Dice 损失用于评估预测结果和真实标签之间的重叠程度，定义如公式所示：
    
    $$ L_{dice}=1−\frac {2∑_{i=1}^Np_ig_i+ϵ} {∑_{i=1}^Np_i^2+∑_{i=1}^Ng_i^2+ϵ} $$
    
    其中，$p_i$ 和 $g_i$ 分别表示第 $i$ 个体素的预测标签和真实标签，$ϵ$ 是一个很小的常数，用于避免分母为零。
    
- **交叉熵损失：**交叉熵损失用于衡量预测概率分布与真实标签分布之间的差异，定义如公式所示：
    
    $$ L_{ce}=−∑_{i=1}^Ng_ilog(p_i) $$
    
- **最终损失函数：** Dice 损失和交叉熵损失的加权组合，如公式所示：
    
    $$ L=L_{dice}+λL_{ce} $$
    
    其中，$λ$ 代表交叉熵损失 $L_{ce}$ 的权重。
    

## 4. 实验

### 4.1. 数据集 (Datasets)

作者在三个不同的3D医学图像分割任务上评估了他们的方法：急性缺血性卒中（AIS）病灶分割、胰腺癌分割和脑肿瘤分割，使用了以下公共数据集：

- **AISD 数据集 (AISD dataset)**
    - **任务：** 卒中病灶分割（Stroke lesion segmentation）。
    - **内容：** 包含397个非增强CT (NCCT) 扫描，描绘了急性缺血性卒中病例，症状发作到CT扫描的时间少于24小时。患者随后在24小时内进行了弥散加权MRI (DWI) 扫描。
    - **扫描参数：** NCCT扫描的切片厚度为5毫米。
    - **分辨率和体积大小范围：** 分辨率范围为 $[0.40–2.04 mm,0.40–2.04 mm,3–11 mm]$，体积大小范围为 $[512, 512, 11-63]$。
    - **数据划分：** 随机选择305个NCCT扫描用于训练，40个用于验证，剩下的52个用于测试。
- **胰腺数据集 (Pancreas dataset)**
    - **任务：** 胰腺癌分割（Pancreas cancer segmentation）。
    - **内容：** 来源于Medical Decathlon的第7个任务，提供了281个带有胰腺和癌症真实标签 (GTs) 的CT扫描。感兴趣区域 (ROIs) 对应于胰腺实质和胰腺肿块（囊肿或肿瘤）。
    - **特点：** 该数据集的挑战在于标签不平衡，包括大（背景）、中（胰腺）和小（肿瘤）结构。
    - **数据来源：** 美国纽约纪念斯隆凯特琳癌症中心 (Memorial Sloan Kettering Cancer Center)。
    - **分辨率和体积大小范围：** 分辨率范围为 $[0.60–0.97 mm,0.60–0.97 mm,0.70–7.5 mm]$，体积大小范围为 $[512, 512, 37–751]$。
    - **数据划分：** 随机选择196例用于训练，28例用于验证，57例用于测试。
- **BraTS2019**
    - **任务：** 脑肿瘤分割（Brain tumor segmentation）。
    - **内容：** 收集了335名脑肿瘤患者的4种模态数据：原始和对比增强T1加权、T2加权和T2液体衰减反转恢复 (T2 FLAIR)。
    - **标注：** 标注包括增强肿瘤 (ET)、水肿 (ED) 和非增强肿瘤核心 (NET)。
    - **实验计算结果的区域：**
        - 整个肿瘤 (WT, 等于 ET+ED+NET)。
        - 增强肿瘤 (ET)。
        - 肿瘤核心 (TC, 等于 ED+NET)。
    - **分辨率和体积大小范围：** 分辨率为 $[1.0 mm,1.0 mm,1.0 mm]$，体积大小范围为 $[240, 240, 155]$。
    - **数据划分：** 仅使用原始训练集，随机选择234例用于训练，34例用于验证，67例用于测试。

### 4.2. 实现细节 (Implementation details)

- **硬件和系统：** 使用PyTorch实现，运行在Ubuntu系统上，配备NVIDIA RTX 3090显卡（24 GB显存）。
- **模型配置：**
    - Stem中的特征通道数设置为8。
    - 训练使用Adam优化器，批量大小 (batch size) 为2，共训练300个周期 (epochs)。
- **优化器设置：**
    - 初始学习率设置为0.0001。
    - 采用poly LR调度，因子 (factor) 默认为0.9。
    - 初始权重衰减 (weight decay) 设置为0.00003。
    - $λ$ 在所有实验中设置为1。
- **数据预处理：**
    - **重采样：** 所有扫描都被重采样到每个数据集的中值分辨率：
        - AISD: $0.4375×0.4375×8.0 mm$
        - Pancreas: $0.8027×0.8027×2.5 mm$
        - BraTS2019: $1.0×1.0×1.0 mm$
    - **插值：** $z$ 轴方向采用最近邻插值 (nearest-neighbor interpolation)，$xy$ 平面采用双线性插值 (bilinear interpolation)。
    - **切块大小 (Patch sizes)：** 训练和推理的切块大小分别为：
        - AISD: $320×320×16$
        - Pancreas: $128×128×64$
        - BraTS2019: $128×128×128$
- **强度归一化：** 为解决数据集内的强度变化，采用 z 分数归一化 (z-score normalization)。
- **数据增强：** 在训练阶段，为提高泛化能力和鲁棒特征捕获，应用了数据增强，包括随机旋转、缩放、翻转和弹性形变 (elastic deformations)。

### 4.3. 对比方法 (Compared methods)

为了验证所提出方法（LW-CTrans）在三个分割任务上的有效性，作者将其与16种最先进 (SOTA) 的基线方法进行了比较。所有对比方法都使用了相同的数据预处理策略。

- **四种 2D 模型：** H2Former, BATFormer, HiFormer, FAT-Net。
- **两种 3D CNN 模型：** AttUNet 3D, nnUNet。
- **一种 3D Transformer 模型：** Dformer。
- **六种 混合模型 (Hybrid models)：** nnFormer, UNETR, CoTr, TransHRNet, PHTrans, U-Mamba。
- **三种 3D 轻量级方法 (3D lightweight methods)：** LCOV-Net, ADHDC-Net, SlimUNETR。

### 4.4. 评估指标 (Evaluation metrics)

- **Dice 分数 (Dice score)**
- **95% Hausdorff 距离 (HD95)**
- **召回率 (recall)**
- **精确率 (precision)**
- **平均对称表面距离 (Average Symmetric Surface Distance, ASSD)**

这些指标提供了对算法分割结果与真实值 (GTs) 之间准确性、完整性和边界一致性的全面评估。

## 5. 结果

### 5.1. 三项任务的比较

- **AISD数据集上的中风病灶分割结果**
    
    - **定量结果 (Quantitative Results):**
        
        ![](https://pic1.imgdb.cn/item/6a27d094edae85a6284f1c9d.png)
        
        - 在AISD（急性缺血性卒中病变数据集）数据集上，LW-CTrans在16个基线方法中表现突出。
        - 它取得了**最佳**的Dice分数（62.35±19.51%）。
        - 取得了**次佳**的95th百分位豪斯多夫距离 (HD95)（40.79±48.76）。
        - 取得了**最佳**的平均对称表面距离 (ASSD)（6.88±10.42）。
    - **观察和分析 (Observations and Analysis):** 3D方法通常优于2D方法，因为它们能更好地捕获中风梗死病灶的空间连续性。nnFormer和UNETR等混合网络的结果不太理想，这归因于它们的主要组成部分仍然是Transformer。
        
    - **视觉结果 (Visual Results):** LW-CTrans的分割结果与中风病灶的真实值（GTs）非常吻合，并且比其他方法产生了更少的假阴性（即漏分的情况更少）。
        ![](https://pic1.imgdb.cn/item/6a27d0bcedae85a6284f1cbf.png)
        
- **胰腺数据集上的胰腺癌分割结果**
    
    - **定量结果 (Quantitative Results)：** 在Pancreas数据集上，LW-CTrans取得了**次佳**的Dice分数（64.69±20.58%），仅略低于nnUNet（65.01±19.05%）。
        ![](https://pic1.imgdb.cn/item/6a27d136edae85a6284f1d15.png)
        
    - **视觉结果 (Visual Results)：** 在胰腺分割上，LW-CTrans的分割结果与真实值非常一致，并且假阴性也较少。
        ![](https://pic1.imgdb.cn/item/6a27d155edae85a6284f1d30.png)
        
- **BraTS2019上的脑肿瘤分割结果**
    
    - **定量结果 (Quantitative Results)：** 在BraTS2019数据集上，LW-CTrans取得了**最高**的Dice分数（83.75±15.77%），比排名第二的U-Mamba高出0.66%（83.09±16.24%）。
        ![](https://pic1.imgdb.cn/item/6a27d171edae85a6284f1d57.png)
        
    - **视觉结果 (Visual Results)：** LW-CTrans的分割结果与脑肿瘤的真实值紧密对齐，并且与其他方法相比，显著减少了**类别间错误**（inter-class error）。
        ![](https://pic1.imgdb.cn/item/6a27d18dedae85a6284f1d7e.png)
        
- **计算复杂度分析：** 该部分将LW-CTrans的性能与计算资源消耗进行权衡比较，主要关注参数数量、浮点运算次数（FLOPs）和模型大小。
    ![](https://pic1.imgdb.cn/item/6a27d1a4edae85a6284f1d8c.png)
    
    - **与SOTA和轻量级模型的对比：**
        - **AISD数据集：** LW-CTrans（参数量2.08M，FLOPs 31G，模型大小15.87 MB）的Dice和ASSD指标优于所有SOTA方法，包括三种轻量级方法（LCOVNet、ADHDC-Net、SlimUNETR）。
        - **Pancreas数据集：** 尽管LW-CTrans的指标略低于nnUNet，但其计算复杂度远优于nnUNet（例如，参数量2.14M vs. 30.75M；FLOPs 83G vs. 710G；模型大小16.33 MB vs. 234.72 MB）。
        - **BraTS2019数据集：** LW-CTrans在参数量2.21M、FLOPs 113G、模型大小16.86 MB的条件下，实现了最佳的平均Dice值（83.75%）。
    - **总结 (Summary):**
        - 相较于非轻量级的3D SOTA方法，LW-CTrans拥有少得多的参数、FLOPs和模型大小。
        - 尽管与LCOVNet、ADHDC-Net和SlimUNETR这三种轻量级方法相比，LW-CTrans的计算复杂度略高，但其分割性能要好得多。

### 5.2. 在三个数据集上进行小目标分割的比较

- **AISD 数据集**
    
    - **实验设置：** 在AISD数据集上，研究人员将CT扫描中的急性缺血性卒中（AIS）病灶体积分为两组：
        
        - 小于 30 mL 的病灶（小病灶）；
        - 大于或等于 30 mL 的病灶（大病灶）。
        - 30 mL 是根据 Liu 等人（2023b）的报告，被设定为病灶体积的中位数。
    - **结果可视化：** 用于直观地显示LW-CTrans与其他对比方法在分割大病灶和小病灶时的 Dice 分布性能。
        ![](https://pic1.imgdb.cn/item/6a27d1dfedae85a6284f1dd2.png)
        
    - **主要发现：**
        
        - 所有方法对大病灶的 Dice 分数都高于对小病灶的 Dice 分数（这是分割任务中的常见挑战）。
        - **LW-CTrans 方法对于大病灶和小病灶都取得了比其他方法更高的性能。**
- **BraTS2019 和 Pancreas 数据集**
    
    - **BraTS2019 数据集：** 对于其中的小目标“增强肿瘤”（ET，Enhanced Tumor），其体积中位数为 13.11 mL。LW-CTrans 在此目标上取得了最高的 Dice 分数，达到 **76.12%**。
    - **Pancreas 数据集：** 对于其中的小目标“胰腺癌”，其体积中位数为 5.78 mL。LW-CTrans 在此目标上也取得了最高的 Dice 分数，达到 **53.19%**。
- **视觉示例：** 展示了LW-CTrans与三种代表性基线方法在上述三个任务中的小目标分割的**视觉示例**。
    ![](https://pic1.imgdb.cn/item/6a27d1f9edae85a6284f1def.png)
    
    - **观察结果：**
        - **在第一行（AISD）**，LW-CTrans 的分割结果与小的卒中病灶的真实值（GTs）非常吻合，并且比其他方法产生了更少的错误分类体素（false classified voxels）。
        - **在第二行（Pancreas）和第三行（BraTS2019）**，LW-CTrans 比其他方法更准确地识别了小的胰腺（可能是指胰腺的某个部分或胰腺癌，但上下文主要强调小目标，结合表2，应指胰腺癌）和小的脑肿瘤（ET）（这些区域通过**白色箭头**指出）。
    - **性能归因：** 这种卓越的性能归因于所提出的 **MVPFormer** 模块的有效性。

### 5.3 消融实验
![](https://pic1.imgdb.cn/item/6a27d238edae85a6284f1e0e.png)

- **动态 Stem 的有效性：** 验证动态 Stem 模块的有效性。动态 Stem 的层数是根据每个数据集的属性自适应设置的。
    - **实验:** 将动态 Stem 替换为一个固定的 Stem。
    - **结果:** 当把 Stem 的层数固定为 2 时，Dice 分数下降了 1.54%。
    - **结论:** 这表明动态 Stem 是有效的，能够根据不同数据集的特性进行适应。
- **MVPFormer 的有效性** 验证我们提出的 MVPFormer 模块的有效性。
    - **实验:** 分别用 PoolFormer 和标准的 Transformer替换 MVPFormer。
    - **结果:** 替换后，Dice 分数分别下降了 2.70% 和 2.19%。
    - **结论:** 这证明了 MVPFormer 是有效的，并且设计能够处理小目标（small targets）的 Transformer 块是必要的。
- **MPConv 块的有效性：** 验证所设计的 MPConv 块的有效性。
    - **实验:** 用 Mobilenetv2 中的 BottleNeck 结构替换第 2 阶段和第 3 阶段的 MPConv 块。
    - **结果:** Dice 分数下降了 2.07%。
    - **结论:** 这表明设计的 MPConv 块是有效的。
- **混合网络（Hybrid network）的有效性**
    - **实验一（移除组件）:** 分别移除第 2 阶段和第 3 阶段的 MVPFormer 块或 MPConv 块。
        - **结果:** 移除 MVPFormer 导致 Dice 下降 1.71%，移除 MPConv 导致 Dice 下降 2.27%。
        - **结论:** 这表明我们设计的混合模块是有效的，并且暗示了 CNN 和 Transformer 的组合是必要的。
    - **实验二（单一架构）:** 进行两项对比实验：
        1. 用 MPConv 替换所有阶段（Stage 1 到 Stage 4）的 MVPFormer 模块。
        2. 用 MVPFormer 替换所有阶段（Stage 1 到 Stage 4）的 MPConv 模块。
    - **结果:**
        - 仅使用 MVPFormer 模块时，Dice 分数下降 3.14%，但参数量略微增加。
        - 仅使用 MPConv 模块时，Dice 分数下降 4.26%，尽管参数量几乎减半。
    - **结论:** 这证明了所提出的混合网络设计是有效的，该设计包括：
        - 在浅层使用卷积块（CNN）进行局部建模。
        - 在中间层使用混合的 CNN-Transformer 块进行局部和全局的同步建模。
        - 在深层使用 Transformer 块进行全局建模。
- **多阶段特征融合（MSFF）的有效性**
    - **实验:** 用一个 CNN 解码器替换 MSFF 模块。该 CNN 解码器类似于 UNet 的解码器，每个阶段包含两个卷积层、批量归一化 (BN) 和 ReLU，并使用转置卷积进行上采样。
    - **结果:** 替换 MSFF 模块后，Dice 分数仅增加了 0.02%，但参数量增加了 4.78 倍。
    - **结论:** 这表明提出的 MSFF 模块是有效的且轻量化的。

## 6. 讨论

轻量化模型的目标是在最大限度地减少参数数量的同时，保持高性能。在这项工作中，研究人员首先采用了一个**动态 Stem**模块来适应各种输入图像尺寸。随后，他们设计了一个**多路径卷积块 (MPConv)用于在早期阶段捕获局部特征，并辅以轻量级 Transformer 块 (MVPFormer)来提取全局特征**。由于该方法由多个不同的模块组成，研究人员进行了一系列实验讨论，以探索最优的模型设计。

### **6.1. Impact of the position of hybrid module (混合模块位置的影响)**

![](https://pic1.imgdb.cn/item/6a27d254edae85a6284f1e1e.png)
在计算机视觉任务中，许多研究表明，深度神经网络在早期阶段更倾向于关注局部特征。随着网络的加深，模型逐渐将焦点转向全局特征。基于这一观察，研究人员采取了以下策略：

- 在**第一阶段**，仅使用 **CNN** 来捕获局部特征。
- 在**中间阶段 (Stage 2 和 Stage 3)**，同时使用 **CNN 和 Transformer** 来捕获局部和全局特征。
- 在**最后阶段**，仅使用 **Transformer** 来捕获全局特征。

为了验证这种方法的有效性，研究人员在不同位置应用了所提出的混合模块。结果（如表 6 所示）表明：

- 仅在 Stage 2 和 Stage 3 使用混合模块时，Dice 分数相比仅在 Stage 2 使用时增加了 2.96%，这表明**混合模块应该用在中间阶段**。
- 在 Stage 1 使用混合模块，Dice 分数相比仅在 Stage 2 使用时下降了 1.61%，这表明**混合模块不应该用在第一阶段**。
- 在所有阶段都使用混合模块，Dice 分数相比仅在 Stage 2 和 Stage 3 使用时下降了 3.25%，这表明**混合模块不应该用在所有阶段**。

### **6.2. Impact of the ratio of the channel split (通道分离比例的影响)**
![](https://pic1.imgdb.cn/item/6a27d273edae85a6284f1e36.png)

为了简化模型，研究人员采用了**通道分离策略**，将输入通道均匀地分成 CNN 通道和 Transformer 通道。为了探究通道分离比例的影响，他们改变了这一比例。

- 默认设置（Stage 2 和 Stage 3 均为 50%:50%）的 Dice 分数为 62.35%。
- 当 Stage 2 设置为 60% CNN : 40% Transformer，Stage 3 设置为 40% CNN : 60% Transformer 时，Dice 分数仅下降了 0.02% (62.33%)。 这表明通过微调通道分离策略，有可能实现增强的分割性能。

### **6.3. Impact of the number of CNN blocks and transformer blocks in the hybrid module (混合模块中 CNN 块和 Transformer 块数量的影响)**
![](https://pic1.imgdb.cn/item/6a27d28aedae85a6284f1e42.png)

为了探究混合模块中 CNN 块和 Transformer 块数量变化的影响，研究人员进行了实验。

- 在默认设置中（Stage 2 为 4 CNN 块和 2 Transformer 块，Stage 3 为 2 CNN 块和 4 Transformer 块），Dice 较高。
- 当将 Hybrid 模块配置为 Stage 2 使用 2 个 CNN 块和 2 个 Transformer 块，Stage 3 使用 4 个 CNN 块和 2 个 Transformer 块时，Dice 分数相比默认设置下降了 1.04%。

这一结果表明，在**初始阶段增加 CNN 块的数量**，并在**后续阶段使用更多的 Transformer 块**，可以获得更好的结果。因此，在这项研究中，研究人员将混合模块配置为：

- **Stage 2：** 4 个 CNN 块和 2 个 Transformer 块。
- **Stage 3：** 2 个 CNN 块和 4 个 Transformer 块。

### 6.4. MVPFormer和MPConv之间交互的影响（Impact of the interaction between MVPFormer and MPConv）

为了研究MVPFormer（Multi-View Pooling Transformer）和MPConv（Multi-path convolution）之间信息交换的影响，作者进行了两项实验，尝试了两种交互策略，这些策略借鉴了ShuffleNet的方法，并在AISD数据集上进行了测试。
![](https://pic1.imgdb.cn/item/6a27d29eedae85a6284f1e55.png)

1. **交互策略一（Interaction 1）**: 在Stage 2和Stage 3之间插入了一个通道混洗（channel shuffle）操作。具体来说，它将Stage 2中MPConv和MVPFormer分支的输出通道各交换一半，然后作为Stage 3的输入。
2. **交互策略二（Interaction 2）**: 在Stage 2和Stage 3内部各自插入了一个通道混洗操作。具体来说，在Stage 2中，第一个MVPFormer的输出和第二个MPConv的输出中，有50%的通道被交换；在Stage 3中，第二个MVPFormer的输出和第一个MPConv的输出中，有50%的通道被交换。

**实验结果和推测**: 无论采用交互策略一还是策略二，获得的Dice分数都低于作者提出的默认方法（未加入这些显式交互）。作者推测，这是因为解码器中已经包含了一个点式卷积（Point-wise Convolution），该卷积能够自适应地对编码器中来自MVPFormer和MPConv两个分支的特征进行加权，从而在一定程度上实现了两种特征的交互。因此，在这项研究中，作者没有采用类似于ShuffleNet的MVPFormer和MPConv之间的显式交互。

### 6.5. 网络设计指南（Guidance for network design）

作者根据实验结果提出了设计轻量级3D医学图像分割网络的指导原则：
 
- **混合编码器的有效性：** 设计一个统一的混合编码器比仅使用Transformer作为编码器和CNN作为解码器的方法（如UNETR）或使用并行的CNN和Transformer编码器的方法（如TransHRNet）更有效。
- **分阶段模块选择**:
    - 在**第一阶段（浅层）**，应使用纯粹基于CNN的模块来捕捉局部特征。
    - 在**中间阶段**，应专注于设计有效的CNN和Transformer混合模块，用于同时捕捉局部和全局特征。
    - 在**最后阶段（深层）**，应设计一个纯粹的Transformer模块来捕捉全局特征。
- **轻量级解码器设计：** 对于设计轻量级解码器，应考虑使用简单的**多阶段特征融合（MSFF）**。
- **Stem层的确定**: 数据集属性有助于确定Stem中的层数。

### 6.6. 与基础模型的比较（Comparison with foundation models）

为了比较提出的方法与基础模型（如SAM和SAM-Med2D）的性能，作者在AISD测试集上对每个3D CT进行了实验。
![](https://pic1.imgdb.cn/item/6a27d2afedae85a6284f1e5a.png)

- **实验方法**: 基于真实标签，在病灶切片上随机选择5个前景点（Points）或边界框（Bbox）作为提示（prompt），以获得整个3D扫描的分割结果。
- **实验结果：** 无论是使用点提示还是边界框提示，SAM和SAM-Med2D的分割性能都远低于作者提出的方法（62.35±19.51%）。此外，作者提出的方法仅有2.08M参数，并且是**全自动化**的，这使得它在实际临床环境中更具适用性。

### 6.7. 局限性和未来工作（Limitations and future work）

作者指出了当前方法的两个主要局限性：

1. **通道分离策略的简单性**: 当前的通道分离策略（channel split strategy）相对简单。未来计划研究**自适应通道分离策略**。
2. **在Pancreas数据集上的性能**: 提出的方法在Pancreas数据集上未能达到最优分割性能，这主要是因为它优先考虑了**轻量化设计原则**。未来将致力于探索更有效、性能更优越的方法。

## 7. 结论

**核心创新：** 本研究介绍了一种**新颖的轻量级混合网络**，它结合了**卷积神经网络（CNN）和Transformer**的优点，专用于**3D医学图像分割**任务。

**网络结构特点：**

- **分阶段设计轻量级模块：** 针对网络的不同阶段，设计了不同的轻量级模块。
- **MVPFormer（多视角池化Transformer）：** 研究者提出了MVPFormer模块。
    - **功能：** 它通过将3D特征空间投影到三个**2D正交平面**（即**轴向 (Axial)**、**冠状 (Coronal)** 和 **矢状 (Sagittal)** 三个视图）来学习**全局信息**。
    - **目的：** 这种设计特别旨在**准确地分割微小目标（small targets）**，这是传统方法难以解决的挑战。
- **多阶段特征融合解码器：** 使用了**多阶段特征融合**技术来设计一个**轻量级但有效**的解码器，以减少模型参数，同时保持高分割性能。

**实验结果与优势：**

- **性能与复杂度平衡：** 在针对三种不同分割任务（例如，脑卒中病灶、胰腺癌、脑肿瘤）的广泛实验中，结果表明所提出的方法在**性能**和**模型复杂度**之间达到了**最佳平衡**，优于所有其他对比方法。
- **小目标分割能力：** 此外，该网络能够**有效地分割微小目标**。
- **临床价值：** 这种对微小目标的高精度分割能力，可以**帮助医生制定更精确的治疗方案**，具有重要的临床意义。