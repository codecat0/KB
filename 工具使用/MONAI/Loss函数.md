# clDiceLoss

> **clDice – A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation**
> 
> [https://arxiv.org/abs/2003.07311](https://arxiv.org/abs/2003.07311)
> 
> 作者提出了一种 **专门用于保持血管、神经等细长结构拓扑完整性** 的损失函数 —— **clDice**。

## 原理

**背景：**

在医学图像中（如血管、神经、导管），很多目标是细长、管状（tubular）的。常规的 `DiceLoss` 只考虑体素级别的重叠，**不关注拓扑结构是否连接完整**。

一个预测 mask 如果位置稍微偏移，Dice 可能依然很高，但血管被断开，**从临床上是致命错误！**

**核心思想：**

$$clDice = 2 * (T_{prec} * T_{sens}) / (T_{prec} + T_{sens})$$

`**clDice**` 引入了两个结构化拓扑指标：

- $T_{sens}$**（Topological Sensitivity）**：测量预测结构是否完整覆盖了 GT 的骨架。

- $T_{prec}$**（Topological Precision）**：测量 GT 是否完整覆盖了预测结构的骨架。

  

具体定义如下：

1. 首先对`pred mask`和 `GT mask` 做 **细化(skeletonization)**，提取骨架结构。

1. 分别计算骨架是否被另一个掩膜完全覆盖：
    
    $$T_{\text{sensitivity}} = \frac{|S(G) \cap P|}{|S(G)|}, \quad T_{\text{precision}} = \frac{|S(P) \cap G|}{|S(P)|}$$
    
    - $S(G)$：GT的骨架
    
    - $S(P)$：Pred的骨架
    
    - $P$：Pred mask
    
    - $G$：GT mask
    

1. 最终 `**clDice**`：
    
    $$\text{clDice} = \frac{2 \cdot T_{\text{precision}} \cdot T_{\text{sensitivity}}}{T_{\text{precision}} + T_{\text{sensitivity}}}$$
    

**使用场景：**

|场景|是否推荐使用 clDice|
|---|---|
|血管分割（CTA、MRA）|✅ 强烈推荐|
|神经、导管结构|✅ 推荐|
|肿瘤、器官、无结构对象|❌ 不建议，普通 Dice 更合适|
|与 Dice/CrossEntropy 联合使用|✅ 最佳实践：`Loss = Dice + λ * clDice`|

## 示例：

**参数解析：**

|参数名|类型|说明|默认值|
|---|---|---|---|
|`iter_`|`int`|骨架提取时的迭代次数（近似细化）|常用值如 10|
|`smooth`|`float`|平滑项，防止分母为 0，提升数值稳定性|`1e-6`|
|||||

**二分类示例：**

```python
import torch
from monai.losses import SoftclDiceLoss
from monai.losses.dice import one_hot
import torch.nn.functional as F


batch_size = 2
num_classes = 2
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)
pred = F.sigmoid(pred)

# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = SoftclDiceLoss()

# 计算损失
loss = loss_fn(label, pred)
print("SoftclDiceLoss:", loss.item())
```

**多分类示例：**

```python
import torch
from monai.losses import SoftclDiceLoss
from monai.losses.dice import one_hot
import torch.nn.functional as F


batch_size = 2
num_classes = 5
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)
pred = F.softmax(pred, dim=1)  # 应用softmax使其成为概率分布

# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))
label = one_hot(label, num_classes=num_classes)

# 初始化损失函数
loss_fn = SoftclDiceLoss()

# 计算损失
loss = loss_fn(label, pred)
print("SoftclDiceLoss:", loss.item())
```

  

# ContrastiveLoss

> **A simple framework for contrastive learning of visual representations**
> 
> [http://proceedings.mlr.press/v119/chen20j.html](http://proceedings.mlr.press/v119/chen20j.html)
> 
> 作者提出了一种 **对比学习损失函数**，用于自监督表征学习。

## 原理：

**核心思想：**

`**SimCLR**` 的损失是基于如下思想：

1. 给定一批样本，每个样本生成两个视图（数据增强），记为 $(x_i, x_j)$。

1. 使用编码器提取其表示$(z_i, z_j)$，希望：
    
    - $(z_i, z_j)$ 应该靠近（正样本对）
    
    - 与其他样本表示保持远离（负样本对）
    

1. 使用 **温度系数**$\tau$ 调整 softmax 的平滑程度。

`**SimCLR**` 中的损失函数形式为：

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \ne i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$

- $\text{sim}(z_i, z_j)$为余弦相似度

- $\tau$是温度参数（temperature）

- 这个实现采用 `2N × 2N` 相似度矩阵

**适用场景：**

这个损失函数适用于：

- **自监督学习**（如 SimCLR, MoCo）

- **预训练图像编码器**

- **用于医学图像的预训练任务**：如脑区切片、器官视图重建等

- **两个嵌入向量之间对比学习**

## 示例：

```python
import torch
from monai.losses import ContrastiveLoss


batch_size = 2
embedding_dim = 64

pred = torch.randn(batch_size, embedding_dim)
label = torch.randn(batch_size, embedding_dim)

# 初始化损失函数
loss_fn = ContrastiveLoss()

# 计算损失
loss = loss_fn(pred, label)
print("ContrastiveLoss:", loss.item())
```

  

# DiceLoss

## 原理

**Dice 系数定义：**

`**Dice**` **系数**衡量两个集合的相似度，定义为：

$$\text{Dice} = \frac{2 \cdot |A \cap B|}{|A| + |B|}$$

在**图像分割**中，用预测 `P` 和真实标签 `G` 表示：

$$\text{Dice}(P, G) = \frac{2 \sum_i (P_i \cdot G_i)}{\sum_i P_i + \sum_i G_i}$$

**Dice 损失函数：**

$$\text{DiceLoss} = 1 - \text{Dice}(P, G)$$

**特性：**

- **适用于不平衡类别**：比如小肿瘤区域或前循环/后循环区域。

- **连续概率值支持**：不需要对预测进行阈值化。

- **支持多通道（多类别）输入**。

  

**使用场景：**

- **医学图像分割**：尤其是肿瘤、器官等区域较小、类别极不平衡的情况。

- **二分类/多分类分割任务**：适合二值和多通道语义分割。

- **配合其他损失**：常与 `CrossEntropyLoss`、`FocalLoss` 组合使用。

## 示例：

**参数解析：**

|参数名|类型|默认值|说明|
|---|---|---|---|
|`include_background`|`bool`|`True`|是否包含第 0 类（背景）在计算中。若前景很小，建议设为 `False`。|
|`to_onehot_y`|`bool`|`False`|是否将 `target` 转换为 one-hot 形式，类别数从 `input.shape[1]` 推断。|
|`sigmoid`|`bool`|`False`|是否对预测结果应用 `sigmoid`（适用于二分类）。|
|`softmax`|`bool`|`False`|是否对预测结果应用 `softmax`（适用于多分类）。|
|`other_act`|`callable`|`None`|其他激活函数，如 `torch.tanh`；与 `sigmoid/softmax` 互斥。|
|`squared_pred`|`bool`|`False`|是否在分母中使用平方形式，增加数值稳定性。|
|`jaccard`|`bool`|`False`|是否计算 Jaccard Index（即 soft IoU），而不是 Dice。|
|`reduction`|`str`|`"mean"`|输出的聚合方式：`"none"`、`"mean"` 或 `"sum"`。|
|`smooth_nr`|`float`|`1e-5`|为避免分子为 0 添加的小常数。|
|`smooth_dr`|`float`|`1e-5`|为避免分母为 0 添加的小常数。|
|`batch`|`bool`|`False`|是否先在 batch 维度上求和再进行除法，`True` 更稳定。|
|`weight`|`float` or `list` or `torch.Tensor`|`None`|类别权重，支持：标量（统一权重）、列表或张量。|
|`soft_label`|`bool`|`False`|`target` 是否为 soft label（非0/1连续值标签）。|

**二分类示例：**

```python
import torch
from monai.losses import DiceLoss


batch_size = 2
num_classes = 2
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)

# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = DiceLoss(
    include_background=False,  # 不计算背景类的Dice损失
    softmax=False,  # 如果输入是未softmax的概率分布，则设置为True
    to_onehot_y=True,  # 将标签转换为one-hot编码
    sigmoid=True,  # 如果使用sigmoid输出，则设置为True
    reduction='mean',  # 损失的平均值
)

# 计算损失
loss = loss_fn(pred, label)

print("DiceLoss:", loss.item())
```

**多分类示例：**

```python
import torch
from monai.losses import DiceLoss


batch_size = 2
num_classes = 25
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)

# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = DiceLoss(
    include_background=False,  # 不计算背景类的Dice损失
    softmax=True,  # 如果输入是未softmax的概率分布，则设置为True
    to_onehot_y=True,  # 将标签转换为one-hot编码
    sigmoid=False,  # 如果使用sigmoid输出，则设置为True
    reduction='mean',  # 损失的平均值
)

# 计算损失
loss = loss_fn(pred, label)

print("DiceLoss:", loss.item())
```

**DiceCELoss：**

`DiceCELoss = λ × DiceLoss + (1 - λ) × CrossEntropyLoss`

|参数名|类型|默认值|适用于|说明|
|---|---|---|---|---|
|`include_background`|`bool`|`True`|Dice|是否在计算中包含第0类（背景）通道|
|`to_onehot_y`|`bool`|`False`|Dice|是否将标签 `target` 转为 one-hot 格式（类别数由 `input.shape[1]` 推断）|
|`sigmoid`|`bool`|`False`|Dice|是否对输入施加 sigmoid 激活（用于二分类）|
|`softmax`|`bool`|`False`|Dice|是否对输入施加 softmax 激活（用于多分类）|
|`other_act`|`callable`|`None`|Dice|其他自定义激活函数，如 `torch.tanh`|
|`squared_pred`|`bool`|`False`|Dice|是否对预测值和标签平方后再计算分母（稳定性增强）|
|`jaccard`|`bool`|`False`|Dice|若为 `True`，则计算 Jaccard Index (IoU) 而非 Dice|
|`reduction`|`str`|`"mean"`|Dice + CE|损失聚合方式，可选 `"mean"` 或 `"sum"`|
|`smooth_nr`|`float`|`1e-5`|Dice|分子平滑项，避免分子为0|
|`smooth_dr`|`float`|`1e-5`|Dice|分母平滑项，避免除零（nan）|
|`batch`|`bool`|`False`|Dice|若为 True，则在 batch 维度上先求和再计算 Dice|
|`weight`|`Tensor/None`|`None`|Dice + CE|类别权重（CE 用于 class 加权；Dice 用于正例加权）|
|`lambda_dice`|`float`|`1.0`|\-|DiceLoss 的权重|
|`lambda_ce`|`float`|`1.0`|\-|CrossEntropyLoss 的权重|
|`label_smoothing`|`float`|`0.0`|CE|标签平滑值（在0~1之间），用于防止过拟合|

**DiceFocalLoss：**

`DiceFocalLoss = λ * DiceLoss + (1 - λ) * FocalLoss`

|参数名|类型|默认值|说明|
|---|---|---|---|
|`include_background`|`bool`|`True`|是否将第0通道（背景类）纳入损失计算。设为 `False` 可排除背景。|
|`to_onehot_y`|`bool`|`False`|是否将 `target` 转为 one-hot 格式，类别数从 `input.shape[1]` 推断。|
|`sigmoid`|`bool`|`False`|对 `input` 应用 sigmoid 激活，仅用于 `DiceLoss`（FocalLoss 不需要）。|
|`softmax`|`bool`|`False`|对 `input` 应用 softmax 激活，仅用于 `DiceLoss`。|
|`other_act`|`callable`|`None`|指定其他激活函数（如 `torch.tanh`），仅用于 `DiceLoss`。|
|`squared_pred`|`bool`|`False`|是否在 Dice 分母中使用平方版本的 `pred` 和 `target`。|
|`jaccard`|`bool`|`False`|若为 `True`，计算 Jaccard Index（Soft IoU）而非 Dice。|
|`reduction`|`str`|`"mean"`|输出结果的归约方式，可选值：`"none"` / `"mean"` / `"sum"`。|
|`smooth_nr`|`float`|`1e-5`|添加到 Dice 分子中的平滑项，防止为 0。|
|`smooth_dr`|`float`|`1e-5`|添加到 Dice 分母中的平滑项，防止除以 0。|
|`batch`|`bool`|`False`|是否在 batch 维度上聚合交并比（Dice）后再除，以得到整体 Dice。|
|`gamma`|`float`|`2.0`|Focal Loss 中指数项 γ 的值，用于调整聚焦程度。|
|`weight`|`float/list/None`|`None`|每类的权重，若为 list，则其长度应与类别数一致。|
|`lambda_dice`|`float`|`1.0`|Dice Loss 的加权系数，≥ 0。|
|`lambda_focal`|`float`|`1.0`|Focal Loss 的加权系数，≥ 0。|
|`alpha`|`float or None`|`None`|Focal Loss 的 α 平衡系数，∈ \[0, 1\]，若为 None 则不启用 α 平衡。|

  

# GeneralizedDiceLoss

> **Generalized Dice overlap as a deep learning loss function for highly unbalanced segmentations**
> 
> [https://arxiv.org/abs/1707.03237](https://arxiv.org/abs/1707.03237)
> 
> 作者提出 **Generalized Dice Loss（GDL）**，作为一种对类别不平衡更鲁棒的损失函数。

## 原理

**核心思想：**

`Generalized Dice Loss (GDL)` 在此基础上 **引入了每个类别的权重** $w_c$，以降低类别不平衡带来的影响：

$$\text{GDL} = 1 - \frac{2 \sum_c w_c \sum_i p_{ic} g_{ic}}{\sum_c w_c \sum_i (p_{ic} + g_{ic})}$$

其中：

- $c$：类别索引

- $w_c = \frac{1}{(\sum_i g_{ic})^2}$：类别权重，和类别体素数的平方成反比

- $p_{ic}$：预测的概率（softmax 或 sigmoid 输出）

- $g_{ic}$：真实的 one-hot 标签

**使用场景：**

- 多类别医学图像分割（尤其类别非常不平衡）

- 类别较小的器官/肿瘤分割，如：胰腺、肿瘤核心、小血管等

- 希望考虑类别之间权重差异的任务

## 示例：

**参数解析：**

|参数名|类型|默认值|含义说明|
|---|---|---|---|
|`include_background`|`bool`|`True`|是否包含第 0 类（背景）参与计算。设为 `False` 可排除背景类（常用于只评估前景）。|
|`to_onehot_y`|`bool`|`False`|是否将目标标签 `y` 自动转换为 one-hot 格式，类别数量根据 `input.shape[1]` 自动推断。|
|`sigmoid`|`bool`|`False`|是否对预测结果应用 sigmoid 激活（用于二分类）。与 `softmax` 互斥。|
|`softmax`|`bool`|`False`|是否对预测结果应用 softmax 激活（用于多分类）。与 `sigmoid` 互斥。|
|`other_act`|`callable`|`None`|其他激活函数，例如 `torch.tanh`，用户可自定义。若设置此项，将忽略 `sigmoid` 与 `softmax`。|
|`w_type`|`str`|`"square"`|权重计算方式。用于处理类别不平衡。可选项：  <br>- "square"：权重 = 1 / (类体素数)^2  <br>- "simple"：权重 = 1 / 类体素数  <br>- "uniform"：所有类别权重相同|
|`reduction`|`str`|`"mean"`|对 batch 的损失如何聚合：  <br>- "none"：不聚合，返回每个样本的损失  <br>- "mean"：平均值  <br>- "sum"：总和|
|`smooth_nr`|`float`|`1e-5`|加到分子上的平滑项，避免为零。|
|`smooth_dr`|`float`|`1e-5`|加到分母上的平滑项，避免出现 nan。|
|`batch`|`bool`|`False`|是否在 batch 维度上先合并再计算交并比。设为 `True` 可以跨样本共享统计量。|
|`soft_label`|`bool`|`False`|若为 `True`，则标签 `y` 可包含非0/1的 soft label。常用于知识蒸馏或混合标签训练。|

**二分类示例：**

```python
import torch
from monai.losses import GeneralizedDiceLoss


batch_size = 2
num_classes = 2
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)

# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = GeneralizedDiceLoss(
    include_background=False,  # 不计算背景类的Dice损失
    softmax=False,  # 如果输入是未softmax的概率分布，则设置为True
    to_onehot_y=True,  # 将标签转换为one-hot编码
    sigmoid=True,  # 如果使用sigmoid输出，则设置为True
    reduction='mean',  # 损失的平均值
)

# 计算损失
loss = loss_fn(pred, label)

print("GeneralizedDiceLoss:", loss.item())
```

**多分类示例：**

```python
import torch
from monai.losses import GeneralizedDiceLoss


batch_size = 2
num_classes = 25
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)

# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = GeneralizedDiceLoss(
    include_background=False,  # 不计算背景类的Dice损失
    softmax=True,  # 如果输入是未softmax的概率分布，则设置为True
    to_onehot_y=True,  # 将标签转换为one-hot编码
    sigmoid=False,  # 如果使用sigmoid输出，则设置为True
    reduction='mean',  # 损失的平均值
)

# 计算损失
loss = loss_fn(pred, label)

print("GeneralizedDiceLoss:", loss.item())
```

# GeneralizedWassersteinDiceLoss

## 原理

**背景：**

常规的 `Dice Loss`、`CrossEntropy` 等都假设每个类别之间是独立且等价的。但在医学图像中，类别往往存在语义距离（如“肿瘤核心”与“坏死区”更相似，而与“正常组织”差别大）。这就引出了 **Generalized Wasserstein Dice Loss**（GWDL）的设计。

**Wasserstein Distance：**

`Wasserstein` 距离度量的是两个概率分布之间的“搬运代价”。GWDL 结合了这种思想，用一个类别距离矩阵（cost matrix）来引入类间结构信息。

**损失函数公式：**

简化表达如下：

$$\text{GWDL} = 1 - \frac{2 \sum_i \min(y_i, p_i) \cdot (1 - M_{y_i,p_i})}{\sum_i y_i + \sum_i p_i}$$

其中：

- $y_i$：真实标签的 one-hot 向量

- $p_i$：预测的 softmax 概率

- $M$：类别之间的代价矩阵（cost matrix），一般对角线为 0，非对角为 1（或其他语义距离）

**使用场景：**

- 医学图像分割中多类标签的预测任务

- 类别间存在**语义结构/距离信息**，例如脑肿瘤分割、器官亚结构分割等

- 模型在某些类之间常出错，需要引导其理解类别的**相似性和差异性**

  

## 示例：

**参数解析：**

|参数名|类型|默认值|说明|
|---|---|---|---|
|`dist_matrix`|`torch.Tensor` or `np.ndarray` (shape `[C, C]`)|**必填**|类别之间的距离矩阵。对角为 0，其余为语义或欧几里得距离。|
|`weighting_mode`|`{"default", "GDL"}`|`"default"`|决定类别错误加权方式。  <br>🔹"default"：原始论文方法，推荐。  <br>🔹"GDL"：仿 GDL 方法，来自 COVID-19 分割研究。|
|`reduction`|`{"none", "mean", "sum"}`|`"mean"`|结果的汇总方式：  <br>🔹"none"：返回每个样本的损失  <br>🔹"mean"：返回所有损失的平均值  <br>🔹"sum"：返回总和|
|`smooth_nr`|`float`|`1e-5`|避免分子为 0 的平滑项（用于数值稳定性）|
|`smooth_dr`|`float`|`1e-5`|避免分母为 0 的平滑项（用于数值稳定性）|

**多分类示例：**

```python
import torch
from monai.losses import GeneralizedWassersteinDiceLoss


batch_size = 2
num_classes = 5
height, width, depth = 32, 32, 32

# 定义距离矩阵，表示类别之间的距离，背景类与其他类的距离为4
distance_matrix = torch.tensor([
    [0, 4, 4, 4, 4],
    [4, 0, 1, 2, 3],
    [4, 1, 0, 1, 2],
    [4, 2, 1, 0, 1],
    [4, 3, 2, 1, 0]
], dtype=torch.float32)

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)

# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = GeneralizedWassersteinDiceLoss(
    dist_matrix=distance_matrix
)
# 计算损失
loss = loss_fn(pred, label)
print("GeneralizedWassersteinDiceLoss:", loss.item())
```

# DeepSupervisionLoss

`monai.losses.DeepSupervisionLoss` 是 MONAI 中专为多尺度监督（deep supervision）设计的损失函数封装器。它允许你在训练时利用模型多个输出分支（通常是来自不同尺度或解码阶段）共同参与损失计算，从而帮助模型更快收敛并获得更好的分割性能。

## 示例：

**参数解析：**

|参数名|类型|默认值|说明|
|---|---|---|---|
|`loss`|`nn.Module`|必须提供|用于每个分支的基础损失函数，例如 `DiceLoss()`、`DiceCELoss()` 等|
|`weight_mode`|`str`|`"exp"`|权重计算方式，用于不同尺度的输出分支，决定各分支损失的相对重要性|
|`weights`|`List[float]`|`None`|手动设置的权重列表，若提供，则优先使用而忽略 `weight_mode` 的设置|

|模式名|说明|示例（4个输出分支）|
|---|---|---|
|`"same"`|所有输出分支权重一致|`[1.0, 1.0, 1.0, 1.0]`|
|`"exp"`|指数衰减权重（每个下层分支乘 0.5）|`[1.0, 0.5, 0.25, 0.125]`|
|`"two"`|第一个为 1，其他都为 0.5（通常用于主分支权重大，其余参与少量监督）|`[1.0, 0.5, 0.5, 0.5]`|
|自定义|使用 `weights=[...]` 传入时，`weight_mode` 将被忽略，直接使用权重。|例如：`[0.6, 0.3, 0.1]`|

**二分类示例：**

```python
import torch
from monai.losses import DiceLoss, DeepSupervisionLoss


batch_size = 2
num_classes = 2
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)
preds = [pred] * 3  # 模拟深度监督的多层输出

# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = DeepSupervisionLoss(
    loss=DiceLoss(
        include_background=False,
        to_onehot_y=True,
        softmax=False,
        sigmoid=True,
    ),
    weights=[1.0, 0.5, 0.25]
)
# 计算损失
loss = loss_fn(preds, label)
print("DeepSupervisionLoss:", loss.item())
```

**多分类示例：**

```python
import torch
from monai.losses import DiceLoss, DeepSupervisionLoss


batch_size = 2
num_classes = 25
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)
preds = [pred] * 3  # 模拟深度监督的多层输出

# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = DeepSupervisionLoss(
    loss=DiceLoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        sigmoid=False,
    ),
    weights=[1.0, 0.5, 0.25]
)
# 计算损失
loss = loss_fn(preds, label)
print("DeepSupervisionLoss:", loss.item())
```

# FocalLoss

### 原理：

**经典交叉熵（CE）损失函数：**

对于单个样本的交叉熵：

$$\text{CE}(p_t) = -\log(p_t)$$

其中：

- $p_t$ 是模型对真实类别的预测概率。

**Focal Loss 公式：**

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- $\alpha_t$：样本平衡因子（可选），解决类别不均衡；

- $(1 - p_t)^\gamma$：**焦点调节因子**，当模型对样本分类越正确（即$p_t \to 1$），该项越小，减轻容易样本的损失权重；

- $\gamma$：调节系数（gamma > 0），常用值为 2；

- $\gamma = 0$ 时，退化为交叉熵损失。

**使用场景：**

- 类别极度不平衡（如医学图像中病灶区域很小）

- 分割任务中背景远多于目标区域

- 希望模型更加关注难分类的样本

  

## 示例：

**参数解析：**

|参数名|类型|默认值|说明|
|---|---|---|---|
|`include_background`|`bool`|`True`|是否包含背景通道（即 channel=0）参与损失计算。若设为 `False`，在使用 softmax 的情况下 `alpha` 将无效。|
|`to_onehot_y`|`bool`|`False`|是否将标签 `y` 转换为 one-hot 格式。用于多分类任务。|
|`gamma`|`float`|`2.0`|Focal Loss 中的聚焦参数，控制难分类样本的关注程度。值越大越聚焦于困难样本。|
|`alpha`|`float`|`None`|α-平衡因子，用于缓解类别不平衡问题。范围为 \[0, 1\]。对于二分类，前景设为 α，背景为 1−α。若使用 softmax 且 `include_background=False`，此参数无效。|
|`weight`|`float` 或 `Sequence[float]`|`None`|每类的体素权重。如果为序列，则长度需与类别数一致（若 `include_background=False`，则不包括背景类0）。值不能小于 0。|
|`reduction`|`"none"` / `"mean"` / `"sum"`|`"mean"`|指定如何对最终损失进行归约处理：  <br>- "none"：不归约，输出与输入大小一致  <br>- "mean"：平均损失  <br>- "sum"：总损失和|
|`use_softmax`|`bool`|`False`|是否使用 softmax 将 logits 转换为概率。为 `False` 时使用 sigmoid，适用于二分类或多标签任务。|

**二分类示例：**

```python
import torch
from monai.losses import FocalLoss

batch_size = 2
num_classes = 2
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)

# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = FocalLoss(
    include_background=True,  # 包括背景类
    to_onehot_y=True,  # 将标签转换为one-hot编码
    gamma=2.0,  # 调整难易样本的权重
    alpha=0.25,  # 类别平衡参数
    reduction='mean',  # 平均损失
    use_softmax=False,  # sigmoid or softmax
)
# 计算损失
loss = loss_fn(pred, label)
print("FocalLoss:", loss.item())
```

**多分类示例：**

```python
import torch
from monai.losses import FocalLoss

batch_size = 2
num_classes = 25
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)

# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = FocalLoss(
    include_background=True,  # 包括背景类
    to_onehot_y=True,  # 将标签转换为one-hot编码
    gamma=2.0,  # 调整难易样本的权重
    alpha=0.25,  # 类别平衡参数
    reduction='mean',  # 平均损失
    use_softmax=True,  # sigmoid or softmax
)
# 计算损失
loss = loss_fn(pred, label)
print("FocalLoss:", loss.item())
```

# HausdorffDTLoss

`HausdorffDTLoss` 旨在衡量 **预测分割边界与真实边界之间的最大误差或距离**。

## 原理：

**Hausdorff 距离（HD）** 是两个集合之间的最大最小距离，定义为：

$$HD(A, B) = \max\left\{ \sup_{a \in A} \inf_{b \in B} d(a, b),\; \sup_{b \in B} \inf_{a \in A} d(b, a) \right\}$$

在医学图像分割中：

- $A$：预测的前景边界点集

- $B$：真实的前景边界点集

- $d(a, b)$：通常是欧几里得距离

**使用场景：**

适合用于强调边界精度的 **医学图像分割任务**，如：

- **小器官分割**：如胰腺、肝脏肿瘤、脑血管等。

- **分割目标轮廓清晰、形状重要性高** 的任务。

- **精度要求高的挑战赛或评估指标中包含 HD** 的任务。

  

## 示例：

**参数解析：**

|参数名|类型|默认值|说明|
|---|---|---|---|
|`include_background`|`bool`|`True`|是否包含背景类（索引0）参与计算。如果设为 `False`，背景不参与计算，有助于小目标不被背景信号淹没。|
|`to_onehot_y`|`bool`|`False`|是否将目标 `target` 转为 one-hot 格式，类别数由 `input.shape[1]` 推断。|
|`sigmoid`|`bool`|`False`|是否对预测值应用 sigmoid 激活函数。|
|`softmax`|`bool`|`False`|是否对预测值应用 softmax 激活函数。|
|`other_act`|`callable or None`|`None`|其他激活函数，如 `torch.tanh`，用于替代 sigmoid 或 softmax。|
|`reduction`|`str`|`"mean"`|指定输出的归约方式。可选：none：不归约 mean：对输出取平均 sum：对输出求和|
|`batch`|`bool`|`False`|是否在 batch 维度上先对交集与并集进行求和再计算 loss。若为 `False`，loss 是在每个样本上独立计算的。|

**二分类示例：**

```python
import torch
from monai.losses import HausdorffDTLoss


batch_size = 2
num_classes = 2
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)


# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = HausdorffDTLoss(
    include_background=False,  # 是否包含背景类
    to_onehot_y=True,  # 将标签转换为one-hot编码
    sigmoid=True,  
    softmax=False
)

# 计算损失
loss = loss_fn(pred, label)
print("HausdorffDTLoss:", loss.item())
```

**多分类示例：**

```python
import torch
from monai.losses import HausdorffDTLoss


batch_size = 2
num_classes = 25
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)


# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = HausdorffDTLoss(
    include_background=False,  # 是否包含背景类
    to_onehot_y=True,  # 将标签转换为one-hot编码
    sigmoid=False,  
    softmax=True
)

# 计算损失
loss = loss_fn(pred, label)
print("HausdorffDTLoss:", loss.item())
```

# TverskyLoss

## 原理

**背景：**

在语义分割中，常用的 Dice Loss 和 IoU Loss 通常假设正负样本均衡。而在医学图像中，目标区域（如肿瘤）往往非常小，相比背景极其不平衡，Dice Loss 在这种情况下容易受假阳性（False Positive）或假阴性（False Negative）的影响。

为此，**Tversky Loss** 引入两个权重因子 α 和 β，以控制 FP 和 FN 的相对惩罚力度。

**数学定义：**

**Tversky Index（T.I.）**的定义如下：

$$TI = \frac{TP}{TP + \alpha \cdot FP + \beta \cdot FN}$$

- **TP**：True Positives（预测为正且真实为正）

- **FP**：False Positives（预测为正但真实为负）

- **FN**：False Negatives（预测为负但真实为正）

- **α, β**：可调参数，α + β = 1 通常是常用设定（但不是必须）

当 α = β = 0.5 时，`Tversky Index` 就是 `Dice` 系数：

$$TI = \frac{2TP}{2TP + FP + FN} = \text{Dice Coefficient}$$

**Tversky Loss** 是 1 减去 `Tversky Index`：

$$\mathcal{L}_{Tversky} = 1 - TI$$

**调整 α, β 控制偏好：**

|应用场景|α 值|β 值|含义|
|---|---|---|---|
|偏好减少 **FP**|0.7|0.3|更关注精度（precision）|
|偏好减少 **FN**|0.3|0.7|更关注召回率（recall）|
|训练稳定、折中|0.5|0.5|相当于 Dice Loss|

## 示例：

|参数名|类型|默认值|说明|
|---|---|---|---|
|`include_background`|`bool`|`True`|是否包含第 0 通道（背景）参与损失计算。若为 `False`，则排除背景类别。|
|`to_onehot_y`|`bool`|`False`|是否将目标 `y` 转为 one-hot 格式（用于多类分割任务）。|
|`sigmoid`|`bool`|`False`|若为 `True`，对预测值应用 `sigmoid`（用于二分类）。|
|`softmax`|`bool`|`False`|若为 `True`，对预测值应用 `softmax`（用于多类分类）。|
|`other_act`|`callable` 或 `None`|`None`|自定义激活函数，例如 `torch.tanh`；若设置，`sigmoid` 和 `softmax` 会被忽略。|
|`alpha`|`float`|`0.5`|假阳性（FP）的惩罚系数。|
|`beta`|`float`|`0.5`|假阴性（FN）的惩罚系数。|
|`reduction`|`str`|`"mean"`|损失的聚合方式：`"none"`、`"mean"` 或 `"sum"`。|
|`smooth_nr`|`float`|`1e-5`|加入到分子，避免为 0，提升数值稳定性。|
|`smooth_dr`|`float`|`1e-5`|加入到分母，避免除以 0。|
|`batch`|`bool`|`False`|是否在 batch 维度上进行总和再计算 loss。若为 `True`，则先在 batch 上求和再除；否则每个样本独立计算。|
|`soft_label`|`bool`|`False`|若为 `True`，表示标签为 soft label（如概率分布），而不是严格的二值标签。|

**二分类示例：**

```python
import torch
from monai.losses import TverskyLoss


batch_size = 2
num_classes = 2
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)


# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = TverskyLoss(
    include_background=False,  # 是否包含背景类
    to_onehot_y=True,  # 将标签转换为one-hot编码
    sigmoid=True,
    softmax=False,
    alpha=0.7,  # Tversky损失的alpha参数
    beta=0.3,  # Tversky损失的beta参数
)

# 计算损失
loss = loss_fn(pred, label)
print("TverskyLoss:", loss.item())
```

**多分类示例：**

```python
import torch
from monai.losses import TverskyLoss


batch_size = 2
num_classes = 25
height, width, depth = 32, 32, 32

# 随机生成预测输出（未softmax）
pred = torch.randn(batch_size, num_classes, height, width, depth)


# 随机生成标签（int型类别标签，非one-hot）
label = torch.randint(0, num_classes, (batch_size, 1, height, width, depth))

# 初始化损失函数
loss_fn = TverskyLoss(
    include_background=False,  # 是否包含背景类
    to_onehot_y=True,  # 将标签转换为one-hot编码
    sigmoid=False,
    softmax=True,
    alpha=0.7,  # Tversky损失的alpha参数
    beta=0.3,  # Tversky损失的beta参数
)

# 计算损失
loss = loss_fn(pred, label)
print("TverskyLoss:", loss.item())
```