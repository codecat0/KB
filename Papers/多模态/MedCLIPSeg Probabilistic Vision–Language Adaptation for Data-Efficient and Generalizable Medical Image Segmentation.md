---
Date: 2026-06-10
tags:
  - 分割
  - 多模态大模型
  - 乳腺超声
  - 息肉内镜
  - 脑肿瘤
  - 皮肤镜
来源: CVPR
年份: "2026"
机构: 康考迪亚大学
Code: https://github.com/HealthX-Lab/MedCLIPSeg
---
## 一、简述

### 1. 研究背景

医学图像分割是临床诊断、治疗规划和定量随访的重要基础。然而，该领域长期面临三大核心挑战：

1. **标注数据稀缺**：专家级像素级标注成本高昂，且不同标注者之间往往存在不一致性，导致高质量监督数据严重不足，限制了传统监督学习的性能。

2. **解剖结构模糊性**：病灶和器官边界常因强度渐变、部分容积效应等因素呈现模糊特征，难以做出明确决策，传统确定性模型容易产生过自信（over-confident）预测，无法提供可靠性警示。

3. **域偏移（Domain Shift）**：不同扫描仪、采集协议和患者群体导致的分布差异，使得在有限的域内（In-Distribution, ID）数据上训练的模型，在域外（Out-of-Distribution, OOD）场景下性能急剧下降。

传统方法主要依赖纯视觉模型，如 U-Net 及其变体、TransUNet、Swin-UNet 等。这些模型虽在特定数据集上取得进展，但高度依赖大量标注数据，且大多为确定性输出，缺乏对不确定性的显式建模，在泛化性和临床可信度上存在局限。

近年来，视觉-语言模型（VLMs）如 CLIP 及其生物医学变体（BiomedCLIP、UniMedCLIP 等）通过大规模图像-文本对比预训练，提供了强大的跨模态对齐能力。这为**文本驱动（text-guided）的医学图像分割**开辟了新路径：临床描述比像素级掩码更容易获取，在低数据场景下可通过文本监督补偿标注不足，并支持自然语言交互。然而，现有的 CLIP 适配方法在医学领域仍面临挑战：医学图像视觉差异细微、边界模糊，导致密集预测（dense prediction）能力较弱；多数方法采用单向融合或确定性表示，易在 OOD 数据上过自信；参数高效适配与不确定性感知的结合仍未得到充分探索。

### 2. 本文贡献

针对上述问题，本文提出 **MedCLIPSeg**，一种概率视觉-语言适配框架，用于实现**数据高效、泛化性强且不确定性感知**的文本驱动医学图像分割。主要贡献如下：

1. **双向表征级融合（Bidirectional Representation-level Fusion）**：提出 Probabilistic Vision-Language (PVL) Adapter，在 CLIP 多个深度编码层中实现图像 patch tokens 与文本 tokens 的双向交互，同时保持 CLIP 预训练参数冻结（参数高效）。结合软 patch 级对比损失（soft patch-level contrastive loss），提升低监督下的细粒度语义对齐和数据效率。

2. **概率跨模态注意力（Probabilistic Cross-modal Attention）**：通过变分建模 Key 和 Value 的均值与方差，实现置信度加权注意力，有效捕捉 aleatoric（数据固有模糊性）和 epistemic（模型对未见域的）不确定性，降低过自信，提升准确性和泛化能力。

3. **像素级不确定性图（Pixel-level Uncertainty Maps）**：通过 Monte Carlo 采样 Value 分布，同时输出分割掩码和基于熵的不确定性图，为临床医生提供直观的局部可靠性可视化，支持更安全的决策。

4. **全面实验验证**：在跨越 5 种成像模态、6 个器官/组织的 **16 个公开数据集** 上进行广泛评估，涵盖数据高效性（10%-100% 数据比例）、域泛化（OOD 跨数据集）、不确定性校准等多维度。MedCLIPSeg 在准确性、鲁棒性和可靠性上全面超越现有 SOTA 方法（包括纯视觉模型和 CLIP 适配方法），并通过消融实验验证各组件的有效性。

总体而言，本文首次将概率化视觉-语言建模系统性地应用于医学图像分割，展示了其在提升数据效率、跨域鲁棒性和临床可解释性方面的潜力，为未来文本交互式医疗 AI 系统奠定了基础。


## 二、方法

![](https://pic1.imgdb.cn/item/6a28c409edae85a628527a7b.png)

### 1 CLIP 概述
`MedCLIPSeg` 构建在 `CLIP` 架构之上。`CLIP` 有两个 `Transformer` 编码器：
- **视觉编码器** $E_v$
- **文本编码器** $E_t$

它们将图像和文本映射到共享的 $D$-维嵌入空间，实现跨模态对齐。

**视觉分支**：输入图像批次 $\mathbf{X}_v \in \mathbb{R}^{B \times 3 \times H \times W}$（ $B$ 是批大小）。图像被切分成  $P$ 个非重叠 `patch`，每个 `patch` 投影为 $D$-维嵌入，并添加一个可学习的 `[CLS] token`。最终得到：
$$
\mathbf{Z}_v = E_v(\mathbf{X}_v) \in \mathbb{R}^{B \times (P+1) \times D} \tag{1}
$$
- $\mathbf{Z}_v$ 中的 `patch tokens` 保留空间细节，`[CLS] token` 作为全局图像表征。
- 原理：`CLIP` 预训练虽只对齐全局 `[CLS]`，但 `patch tokens` 已隐含语义空间信息，可用于密集预测。

**文本分支**：文本提示 $\mathbf{X}_t \in \mathbb{R}^{B \times L}$（$L$ 为 token 长度）经嵌入和编码器处理，取 `[EOS] token` 作为全局文本表征：
$$
\mathbf{Z}_t = E_t(\mathbf{X}_t) \in \mathbb{R}^{B \times L \times D} \tag{2}
$$
- $\mathbf{Z}_t[\text{EOS}]$ 用于后续与视觉 `patch` 计算相似度，生成分割 `logit`。

**在框架中的作用**：全局文本嵌入作为 `query`，与视觉 `patch tokens` 做点积，指导文本驱动的分割。这利用了 `CLIP` 的零样本能力，但需进一步适配以适应医学图像的模糊性和域偏移。

### 2 概率多模态适配（核心创新：PVL Adapter）
![](https://pic1.imgdb.cn/item/6a28d336edae85a628530056.png)
这是方法的最重要部分。作者提出 **Probabilistic Vision-Language (PVL) Adapter**，在 CLIP 多个深层之间进行双向、概率化的图像-文本融合，实现置信度加权注意力，并显式建模不确定性。PVL Adapter 保持 CLIP 预训练参数冻结，仅训练适配器（参数高效）。

**向下投影（Downward Projection）**：
给定第 $n$ 层的视觉 `tokens` $\mathbf{V}_v^n$ 和文本 `tokens` $\mathbf{T}_t^n$，先投影到低维共享空间 $D_s$（降低计算量）：
$$
\mathbf{v}^n = \mathbf{V}_v^n \mathbf{W}_v^s, \quad \mathbf{t}^n = \mathbf{T}_t^n \mathbf{W}_t^s \tag{3}
$$
- $\mathbf{W}_v^s, \mathbf{W}_t^s$ 是可学习投影矩阵。

**QKV 参数化（概率化注意力核心）**：
受变分推理启发，将标准注意力扩展为概率形式。对 `Query`、`Key`、`Value` 建模均值和方差（捕捉不确定性）。

输入 query 序列 $\mathbf{X}$，上下文序列 $\mathbf{Z}$，投影：
$$
[\mathbf{Q}] = \mathbf{X} \mathbf{W}_Q \tag{4}
$$
$$
[\mathbf{K}_\mu, \mathbf{K}_{\log\sigma^2}] = \mathbf{Z} \mathbf{W}_K \tag{5}
$$
$$
[\mathbf{V}_\mu, \mathbf{V}_{\log\sigma^2}] = \mathbf{Z} \mathbf{W}_V
$$
- $\mathbf{K}_\mu, \mathbf{V}_\mu$: 均值（deterministic 部分）。
- $\mathbf{K}_{\log\sigma^2}, \mathbf{V}_{\log\sigma^2}$: 对数方差（捕捉 aleatoric uncertainty，即数据固有噪声/模糊性，如医学图像的边界模糊）。
- 使用 **softplus** $\zeta(\cdot)$ 将 $\log\sigma^2$ 转为正方差 $\sigma^2$，避免数值不稳定：
$$
\sigma_K^2 = \zeta(\mathbf{K}_{\log\sigma^2}), \quad \sigma_V^2 = \zeta(\mathbf{V}_{\log\sigma^2}) \tag{6}
$$

**置信度加权注意力（Confidence-weighted Attention）**：
传统注意力仅用均值相似度。作者引入方差惩罚，计算概率注意力分数：
$$
S_\mu = \mathbf{Q} \mathbf{K}_\mu^T \tag{7}
$$
$$
S_{\sigma^2} = \mathbf{Q}^2 (\sigma_K^2)^T \tag{8}
$$
- $S_\mu$: 均值相似度（点积）。
- $S_{\sigma^2}$: 方差项，量化 `key` 不确定性对 `query` 的影响（不确定 `token` 被惩罚）。

最终注意力权重（`softmax` 前加置信度惩罚）：
$$
A_{ij} = \frac{\exp(S_{\mu,ij} / \omega_{\mu,ij})}{\sum_r \exp(S_{\mu,ir} / \omega_{\mu,ir})}, \quad \omega_{\mu,ij} = \exp(\beta S_{\sigma^2,ij}) \tag{9}
$$
- $\beta = \frac{2}{\sqrt{3.5}}$（高斯分布半峰全宽相关常数）控制惩罚强度。
- **原理**：不确定 token（高 $\sigma^2$）的注意力权重被动态下调，模型更关注可靠特征。$\beta=0$ 时退化为标准确定性注意力。这提升了 OOD 泛化性和校准性（避免过自信）。

**Value 采样（Value Sampling）**：
对 Value 分布 $\mathcal{N}(\mathbf{V}_\mu, \sigma_V^2)$ 采样（使用重参数化技巧，保证可微）：
$$
\mathbf{V}_{\text{sample}} = \mathbf{V}_\mu + \boldsymbol{\epsilon} \odot \sigma_V, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}) \tag{10,11}
$$
- 训练时单次采样计算输出 $\mathbf{O} = \mathbf{A} \mathbf{V}_{\text{sample}}$。
- 测试时多次 `Monte Carlo` 前向（e.g. 30 次），得到均值分割图和基于熵的不确定性图。
- **原理**：捕捉 `epistemic uncertainty`（模型对未见数据的知识不确定性）。总不确定性 = `aleatoric` + `epistemic`，提供像素级可靠性可视化（临床上非常实用）。

**残差门控（Residual Gating）**：
为稳定早期训练，引入可学习门控：
$$
\mathbf{Y} = g \cdot \mathbf{O}_{\text{out}} + (1 - g) \cdot \mathbf{X}, \quad g = \sigma(\mathbf{W}_{\text{out}} \cdot \text{concat}(\mathbf{X}, \mathbf{O})) \tag{12,13}
$$
-  $g$ 从 ~0.5 开始，逐渐增加对融合特征的依赖。

**双向交互（Bidirectional Interaction）**：
PVL Adapter 双向工作（视觉→文本 和 文本→视觉），在多个 CLIP 层应用：
$$
\hat{\mathbf{V}}^n = \text{Attn}_v^{\text{PVL}}(\mathbf{v}^n, \mathbf{t}^n), \quad \hat{\mathbf{T}}^n = \text{Attn}_t^{\text{PVL}}(\mathbf{t}^n, \mathbf{v}^n) \tag{14,15}
$$
然后向上投影回原始维度并残差连接：
$$
\mathbf{V}^n \leftarrow \mathbf{V}^n + \hat{\mathbf{V}}^n \mathbf{W}_v^n, \quad \mathbf{T}^n \leftarrow \mathbf{T}^n + \hat{\mathbf{T}}^n \mathbf{W}_t^n \tag{16,17}
$$
- **原理**：双向融合让视觉和文本相互精炼，提升跨模态对齐，尤其适合医学中细粒度语义。

### 3 分割头（Segmentation via Pixel-Text Similarity）
最终融合后：
- `L2` 归一化文本 `[EOS]` 嵌入和视觉 `patch tokens`。
- 轻量 `MLP` 将 `[EOS]` 映射为兼容嵌入 $\psi$。
- 计算相似度（点积）+ 双线性上采样得到 `logit`：
$$
\mathbf{M} = \text{Upsample}(\mathbf{V}_{\text{patches}} \cdot \psi(\mathbf{Z}_t[\text{EOS}])) \tag{18,19}
$$
- 使用 `Dice` + `BCE` 损失训练分割。

### 4 软 Patch 级对比损失（Soft Patch-level Contrastive Loss）
为提升数据效率和细粒度对齐，引入软对比损失：
- 对 patch embeddings 平均池化得到区域表征。
- 计算文本-to-图像和图像-to-文本 logit（$\mathbf{P}_{vt}, \mathbf{P}_{tv}$）。
- 用文本相似度生成软标签 $\mathbf{G} = \text{softmax}(\cdot / \tau)$，$\tau=0.2$。
- 软交叉熵损失，双向平均：
$$
\mathcal{L}_{\text{SoftCon}} = \frac{1}{2} (\mathcal{L}_{\text{soft}}(\mathbf{P}_{vt}, \mathbf{G}) + \mathcal{L}_{\text{soft}}(\mathbf{P}_{tv}, \mathbf{G})) \tag{20-22}
$$

**总体损失**：
$$
\mathcal{L} = \lambda_{\text{Seg}} \mathcal{L}_{\text{Seg}} + \lambda_{\text{SoftCon}} \mathcal{L}_{\text{SoftCon}} \tag{23}
$$
$( \lambda_{\text{Seg}}=0.5$, $\lambda_{\text{SoftCon}}=0.1$）。

### 方法总结与创新点
- **概率建模**：通过变分 Key/Value + 置信度加权 + MC 采样，实现不确定性感知，解决医学图像模糊性和域偏移。
- **双向深层融合**：多层 PVL Adapter + 残差门控，提升数据效率和泛化。
- **软对比**：支持多样文本提示下的细粒度对齐。
- **优势**：冻结 CLIP 主干，参数高效；输出不确定性图，便于临床信任；在大规模多模态、多器官实验中优于 SOTA。
## 三、实验结果

### 1 实验设置与数据集
论文在**16个公开医学图像分割数据集**上进行了全面评估，覆盖**5种成像模态**和**6个器官/组织**，强调数据高效性（低标注数据）和域泛化（OOD跨数据集/设备/协议）。

**主要数据集**：
- **超声（Ultrasound）**：BUSI、BUSBRA、BUSUC、BUID、UDIAT 等（乳腺等）。
- **内镜/结肠（Endoscopy/Polyp）**：Kvasir-SEG、CVC-ColonDB、CVC-ClinicDB、CVC-300、BKAI 等。
- **皮肤病（Dermatoscopy/Skin Lesion）**：ISIC、UWaterlooSkinCancer。
- **其他**：脑MRI（BTMRI）、EUS 等，以及 QaTa-COV19 等。
- 对于缺少文本描述的数据集，使用 GPT-5 生成临床风格提示，并结合图像处理辅助。

**评估设置**：
- **数据高效性**：使用 10%、25%、50% 和 100% 训练数据训练，测试全数据性能（DSC + NSD 指标）。
- **域泛化（OOD）**：在一个源数据集上全监督训练，直接零样本测试到其他未见目标数据集（跨设备、协议、人口统计学偏移）。
- **基线**：对比 U-Net 系列、TransUNet、Swin-UNet 等视觉模型，以及 LViT、CLIPSeg、DenseCLIP、CAT-Seg、MaPLe、VLSM-Adapter、CausalCLIPSeg 等文本驱动/CLIP 适配方法。
- **骨干**：UniMedCLIP ViT-B/16 + PubMedBERT。
- **指标**：Dice Similarity Coefficient (DSC)、Normalized Surface Distance (NSD)；额外评估不确定性校准（Brier 分数等）。

### 2 主要实验结果
![](https://pic1.imgdb.cn/item/6a28d7bbedae85a6285301be.png)

**1. 数据高效性**：
![](https://pic1.imgdb.cn/item/6a28d711edae85a62853019c.png)
- MedCLIPSeg 在所有数据比例下均显著优于基线，尤其在**极低数据 regime**（10% 数据）下优势明显（相比强基线提升 2-3% DSC，相比某些 CLIP 方法提升更高）。
- 随着数据增加，性能稳步提升，在全数据下达到最佳。
- 软对比损失（SoftCon）和 PVL Adapter 显著提升低数据下的细粒度对齐和泛化。

**2. 域泛化**：
![](https://pic1.imgdb.cn/item/6a28d73cedae85a6285301a7.png)
- 在多个跨数据集场景中表现出色，例如：
  - 乳腺超声：BUSI 等目标上 DSC 达 **85.72%**。
  - 其他如 BUSUC **84.37%**、Kvasir-SEG **90.15%**、BTMRI **88.03%**、ISIC **92.54%** 等（部分高分）。
- 优于 CAT-Seg、SAN 等 SOTA，证明概率双向融合有效缓解域偏移，降低 OOD 性能下降，并提供更好置信度校准（更低 Brier 分数）。

**3. 消融研究**：
![](https://pic1.imgdb.cn/item/6a28d7e2edae85a6285301c4.png)
![](https://pic1.imgdb.cn/item/6a28d817edae85a6285301d7.png)
- PVL Adapter（概率注意力）、双向交互、残差门控、Soft Patch-level Contrastive Loss 等组件均带来稳定提升。
- 概率建模特别提升不确定性感知和 OOD 鲁棒性。
- 不确定性图（MC 采样 + 熵）能有效突出分割边界模糊或低置信区域，具有临床解释性。

**总体结论**：
MedCLIPSeg 在**准确性、数据效率、跨域鲁棒性**上全面超越先前方法，同时通过像素级不确定性图提升临床可信度。实验规模大（多模态、多器官）、设置严谨（ID/OOD、低数据），充分验证了概率视觉-语言适配在医学分割中的潜力。

