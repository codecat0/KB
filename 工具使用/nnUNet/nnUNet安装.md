## 1. **创建并激活 Conda 环境**

创建一个独立的 Conda 环境以隔离 `nnU-Net` 依赖：

```bash
conda create -n nnunet python=3.102
conda activate nnunet
```

## 2. **安装 PyTorch**

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

## 3. **克隆 nnU-Net 仓库**

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
```

## 4. **安装 nnU-Net**

在 `nnU-Net` 目录下以可编辑模式安装：

```bash
cd /path/to/nnUNet
pip install -e .
```

说明：`-e` 表示可编辑安装，便于开发或修改 nnU-Net 代码。

  

验证安装：

```bash
nnUNetv2_plan_and_preprocess --help
```

![[image 36.webp|image 36.png]]

  

## 5. **创建必要目录**

为 `nnU-Net` 创建数据和结果目录：

```bash
cd /path/to/nnUNet
mkdir nnUNet_raw
mkdir nnUNet_preprocessed
mkdir nnUNet_results
```

- `nnUNet_raw`：存储原始数据集。

- `nnUNet_preprocessed`：存储预处理后的数据。

- `nnUNet_results`：存储训练模型和推理结果。

  

## 6. **配置环境变量**

`nnU-Net` 需要环境变量指向数据目录。找到您目录中的 `.bashrc` 文件，并在底部添加以下行：

```bash
export nnUNet_raw="/path/to/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet/nnUNet_results"
```

![[image 1 30.webp|image 1 30.png]]

  

验证是否设置成功：`echo ${nnUNet_raw}`

![[image 2 30.webp|image 2 30.png]]