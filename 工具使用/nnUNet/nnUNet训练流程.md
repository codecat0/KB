  

## 1. 准备数据集

### nnUNet格式数据集生成：

`nnUNet` 要求数据集遵循特定结构，存储在 `NNUNET_RAW` 目录（默认 `/path/to/nnUNet_raw`）。数据集需按以下格式组织：

```yaml
nnUNet_raw/DatasetXXX_Name/
├── dataset.json
├── imagesTr/
│   ├── case_000_0000.nii.gz
│   ├── case_001_0000.nii.gz
│   ├── case_002_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── case_000.nii.gz
│   ├── case_001.nii.gz
│   └── ...
├── imagesTs/
│   ├── case_100_0000.nii.gz
│   └── ...
```

- `**DatasetXXX_Name**`：数据集名称，`XXX` 是一个三位数字 ID（如 `Dataset001_Liver`）。

- `**dataset.json**`：描述数据集的元信息，包括模态、标签等。

- `**imagesTr**`：训练图像（`.nii.gz` 格式），多模态图像以 `_0000`、`_0001` 等后缀区分。

- `**labelsTr**`：训练标签（分割掩膜）。

- `**imagesTs**`：测试图像（可选）。

  

`**XXX**` **是一个三位数字 ID，这个数字ID十分重要，用于数据集识别**

  

下方`nnunet_dataset.py`脚本提供将数据集转换为`nnUNet`格式：

- `**src_dir**`：训练数据集路径，其中包含两个文件夹`images`和`labels`，分别存放**图像**和**标签**

- `**tgt_dir**`：转换后数据集存放路径，其名称需以`**DatasetXXX_Name**`，其中`XXX`为**数字ID**，`Name`为**任务名称**

- `**task_name**`：同上**任务名称**，转换后的的文件名会以`{task_name}_xxx_0000.nii.gz`命名

- `**test_dir**`：测试数据集路径，其中包含`images`文件夹，存放**图像**

```python
# This script is designed to convert a dataset into the nnUNet format.
# It processes images and labels, ensuring they are in the correct format and shape.
# The script uses multithreading to speed up the processing of multiple files.
# It also includes error handling to log any issues encountered during processing. 

import argparse
import os
import glob
import random
import traceback
import threadpool
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from loguru import logger
from collections import defaultdict

succeed = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir", type=str, required=True, help="Source directory containing the images and labels"
    )
    parser.add_argument(
        "--tgt_dir", type=str, required=True, default="Dataset001_ASPECTS", help="Target directory to save the processed dataset"
    )
    parser.add_argument(
        "--task_name", type=str, required=True, default='ASPECTS', help="Name of the task for nnUNet dataset format"
    ) 
    parser.add_argument(
        "--test_dir", type=str, default=None, help="Directory for test data, if any. Not used in this script but can be extended later."
    )
    args = parser.parse_args()
    return args


def load_scan(sitk_path):
    try:
        sitk_image = sitk.ReadImage(sitk_path)
        vol_arr = sitk.GetArrayFromImage(sitk_image)
        vol_shape = vol_arr.shape
    except Exception as e:
        logger.error(f"Error loading {sitk_path}: {e}")
        return None, None, None
    return sitk_image, vol_arr, vol_shape



def gen_single_data(info):
    global succeed
    (
        formatted_num,
        vol_file, 
        anno_file, 
        tgt_img_file, 
        tgt_anno_file, 
    ) = info
    try:
        sitk_image, _, vol_shape = load_scan(vol_file)
        _, anno_arr, anno_shape = load_scan(anno_file)
        anno_arr = anno_arr.astype(np.uint8)

        if vol_shape != anno_shape:
            logger.error(f"Shape mismatch for {vol_file} and {anno_file}: {vol_shape} vs {anno_shape}")
            raise ValueError(f"Shape mismatch for {vol_file} and {anno_file}")
        
        sitk.WriteImage(sitk_image, tgt_img_file)

        sitk_anno = sitk.GetImageFromArray(anno_arr)
        sitk_anno.CopyInformation(sitk_image)  # Copy metadata from the image
        sitk.WriteImage(sitk_anno, tgt_anno_file)
        succeed += 1
    except Exception as e:
        logger.error(f"Error processing {vol_file}, {anno_file}: {e}")
        traceback.print_exc()


def main():
    args = parse_args()
    vol_dir = os.path.join(args.src_dir, "images")
    anno_dir = os.path.join(args.src_dir, "labels")

    tgt_dir = os.path.join(os.environ.get('nnUNet_raw'), args.tgt_dir)
    os.makedirs(tgt_dir, exist_ok=True)
    tgt_img_dir = os.path.join(tgt_dir, "imagesTr")
    tgt_anno_dir = os.path.join(tgt_dir, "labelsTr")
    os.makedirs(tgt_img_dir, exist_ok=True)
    os.makedirs(tgt_anno_dir, exist_ok=True)

    vol_lst = sorted(glob.glob(os.path.join(vol_dir, "*.nii.gz")))
    anno_lst = sorted(glob.glob(os.path.join(anno_dir, "*.nii.gz")))
    if len(vol_lst) != len(anno_lst):
        logger.error(f"Number of volume files ({len(vol_lst)}) does not match number of annotation files ({len(anno_lst)})")
        return
    idx = 0
    data_lst = []
    for i in range(len(vol_lst)):
        try:
            vol_file = os.path.join(vol_dir, vol_lst[i])
            anno_file = os.path.join(anno_dir, anno_lst[i])
            if not os.path.exists(anno_file):
                logger.error(f"Annotation file {anno_file} does not exist, skipping")
                continue
            if not os.path.exists(vol_file):
                logger.error(f"Volume file {vol_file} does not exist, skipping")
                continue
            formatted_num = str(idx).zfill(3)
            tgt_img_file = os.path.join(tgt_img_dir, f"{args.task_name}_{formatted_num}_0000.nii.gz")
            tgt_anno_file = os.path.join(tgt_anno_dir, f"{args.task_name}_{formatted_num}.nii.gz")
            data_lst.append([
                formatted_num, 
                vol_file, 
                anno_file, 
                tgt_img_file, 
                tgt_anno_file
            ])
            idx += 1
        except Exception as e:
            logger.error(f"Error processing {vol_file} or {anno_file}: {e}")
            traceback.print_exc()
        
    logger.info(f"Total {len(data_lst)} data to process")
    pool = threadpool.ThreadPool(16)
    requests = threadpool.makeRequests(gen_single_data, data_lst)
    for req in requests:
        pool.putRequest(req)
    pool.wait()
    logger.info(f"Processed {succeed} out of {len(data_lst)} data successfully")
    logger.info(f"Dataset saved to {tgt_dir}")
    
    if args.test_dir:
        test_dir = os.path.join(args.test_dir, "images")
        tgt_test_dir = os.path.join(tgt_dir, "imagesTs")
        os.makedirs(tgt_test_dir, exist_ok=True)
        test_lst = sorted(glob.glob(os.path.join(test_dir, "*.nii.gz")))
        for i, test_file in enumerate(test_lst):
            try:
                formatted_num = str(i).zfill(3)
                test_file_name = os.path.basename(test_file).replace('.nii.gz', '')
                tgt_test_file = os.path.join(tgt_test_dir, f"{test_file_name}_{formatted_num}_0000.nii.gz")
                sitk_image, _, _ = load_scan(test_file)
                if sitk_image is None:
                    continue
                sitk.WriteImage(sitk_image, tgt_test_file)
            except Exception as e:
                logger.error(f"Error processing test file {test_file}: {e}")
                traceback.print_exc()
        logger.info(f"Test dataset saved to {tgt_test_dir} (if applicable)")


if __name__ == "__main__":
    main()
    if succeed == 0:
        logger.error("No data was successfully processed. Please check the logs for errors.")
    else:
        logger.info("nnUNet dataset generation completed.")
```

  

### dataset.json文件生成：

`nnU-Net` 的 `dataset.json` 是每个任务（Task）数据集的核心配置文件，**必须正确配置**，否则无法进行后续的预处理、训练和推理。  

|字段|说明|举例|备注|
|---|---|---|---|
|`name`|数据集名称|"LiverSeg"|可选|
|`description`|描述|简要介绍|可选|
|`reference`|数据来源（如论文或网站）||可选|
|`licence`|许可|如 CC-BY-SA 4.0|可选|
|`release`|版本|1.0|可选|
|`tensorImageSize`|图像维度|“3D”|可选|
|`channel_names`|输入模态索引和类型|`{ "0": "CT" }`，MRI可能有多个如 `{ "0": "T1", "1": "T2" }`|**必须**|
|`labels`|标签映射（名称→ int）|`{"background": 0, "liver": 1, "tumor": 2}`|**必须**|
|`file_ending`|文件后缀|`.nii.gz`|**必须**|
|`numTraining`|训练样本数|数字（必须和 `training` 数量一致）|**必须**|
|`training`|训练集图像与标签路径列表|每项包含 `"image"` 和 `"label"` 路径|可选|
|`test`|测试集图像路径列表（无标签）|只列 `"image"` 路径即可|可选|

  

下方`generate_dataset_json.py`脚本提供了根据用户输入生成`nnUNet`格式数据集的`dataset.json`文件，用户可在`main`函数修改：

- `**task_name**` ：其名称需以`**DatasetXXX_Name**`，其中`XXX`为**数字ID**，`Name`为**任务名称，**默认`Dataset001_ASPECTS`

- `**name**`：数据集名称

- `**description**`：描述

- `**tensorImageSize**`：图像维度

- `**file_ending**`：文件后缀

- `**channel_names**`：输入模态索引和类型

- `**labels**`：标签映射

```python
import os
import numpy as np
import json
from typing import List

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent, ensure_ascii=False)


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-7] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def generate_dataset_json(output_file: str, 
                          imagesTr_dir: str, 
                          imagesTs_dir: str, 
                          name: str = '', 
                          labels: dict = {}, 
                          description: str = '',
                          tensorImageSize: str = '',
                          reference: str = '',
                          license: str = "",
                          release: str = "",
                          file_ending: str = '.nii.gz',
                          channel_names: dict = None,
                          ):

    # 获取文件夹内各个独立的文件
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)
	  # imagesTs_dir 文件夹可以为空，只要有训练的就行
    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = name
    json_dict['description'] = description
    json_dict['tensorImageSize'] = tensorImageSize
    json_dict['reference'] = reference
    json_dict['license'] = license
    json_dict['release'] = release
    json_dict['labels'] = labels
    if channel_names is not None:
        json_dict['channel_names'] = channel_names
    json_dict['file_ending'] = file_ending
    json_dict['overwrite_image_reader_writer'] = "SimpleITKIO"
	
    json_dict['numTraining'] = len(train_identifiers)
    if imagesTs_dir is not None:
        json_dict['numTest'] = len(test_identifiers)

    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_identifiers]
    if imagesTs_dir is not None:
        json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]
	
    output_file = output_file
    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. Proceeding anyways...")
        
    save_json(json_dict,os.path.join(output_file))


if __name__ == "__main__":
    nnunet_raw = os.environ.get('nnUNet_raw', '../nnUNet_raw')
    task_name = 'Dataset001_ASPECTS'
    output_file = os.path.join(nnunet_raw, task_name, 'dataset.json')
    imagesTr_dir = os.path.join(nnunet_raw, task_name, 'imagesTr')
    labelsTr = os.path.join(nnunet_raw, task_name, 'labelsTr')
    
    name = 'ASPECTS'
    description = 'ASPECTS dataset for ischemic stroke assessment'
    tensorImageSize = '3D'
    file_ending = '.nii.gz'
    channel_names = {
        "0": "CT"
    }
    labels = { 
	    "background": 0, 
	    "左侧尾状核": 1,
	    "左侧豆状核": 2,
        "左侧内囊": 3,
        "左侧M1": 4,
        "左侧M2": 5,
        "左侧M3": 6,
        "左侧M4": 7,
        "左侧M5": 8,
        "左侧M6": 9,
        "左侧岛叶": 10, 
        "右侧尾状核": 11,  
        "右侧豆状核": 12,
        "右侧内囊": 13,
        "右侧M1": 14,
        "右侧M2": 15,
        "右侧M3": 16,
        "右侧M4": 17,
        "右侧M5": 18,
        "右侧M6": 19,
        "右侧岛叶": 20,
        "中脑": 21,
        "脑桥": 22,
        "左侧丘脑": 23,
        "右侧丘脑": 24,
        "左侧小脑": 25,
        "右侧小脑": 26,
        "左侧大脑后动脉供血区": 27,
        "右侧大脑后动脉供血区": 28,  
	 
    }
    generate_dataset_json(
        output_file=output_file, 
        imagesTr_dir=imagesTr_dir, 
        imagesTs_dir=None,              # 如果没有测试集，可以设置为 None
        name=name,
        labels=labels,
        description=description,
        tensorImageSize=tensorImageSize,
        channel_names=channel_names,
        file_ending=file_ending,
    )
```

结果如下：

```json
{
    "channel_names": {
        "0": "CT"
    },
    "description": "ASPECTS dataset for ischemic stroke assessment",
    "file_ending": ".nii.gz",
    "labels": {
        "background": 0,
        "中脑": 21,
        "右侧M1": 14,
        "右侧M2": 15,
        "右侧M3": 16,
        "右侧M4": 17,
        "右侧M5": 18,
        "右侧M6": 19,
        "右侧丘脑": 24,
        "右侧内囊": 13,
        "右侧大脑后动脉供血区": 28,
        "右侧小脑": 26,
        "右侧尾状核": 11,
        "右侧岛叶": 20,
        "右侧豆状核": 12,
        "左侧M1": 4,
        "左侧M2": 5,
        "左侧M3": 6,
        "左侧M4": 7,
        "左侧M5": 8,
        "左侧M6": 9,
        "左侧丘脑": 23,
        "左侧内囊": 3,
        "左侧大脑后动脉供血区": 27,
        "左侧小脑": 25,
        "左侧尾状核": 1,
        "左侧岛叶": 10,
        "左侧豆状核": 2,
        "脑桥": 22
    },
    "license": "",
    "name": "ASPECTS",
    "numTraining": 120,
    "overwrite_image_reader_writer": "SimpleITKIO",
    "reference": "",
    "release": "",
    "tensorImageSize": "3D",
    "training": [
        {
            "image": "./imagesTr/ASPECTS_000_0000.nii.gz",
            "label": "./labelsTr/ASPECTS_000_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ASPECTS_001_0000.nii.gz",
            "label": "./labelsTr/ASPECTS_001_0000.nii.gz"
        },
        ...
     ]
  }
```

  

## 2. 数据预处理

### 核心命令格式：

`nnUNet` 的预处理包括归一化、裁剪和重采样，自动优化数据以适配模型。运行以下命令：

```bash
nnUNetv2_plan_and_preprocess -d XXX --verify_dataset_integrity
```

- `d XXX`：指定数据集 ID。

- `-verify_dataset_integrity`：验证数据完整性。

### 参数详解：

`**nnUNetv2_plan_and_preprocess**`参数如下：

|参数|含义|示例|
|---|---|---|
|`-d D [D ...]`|**[必须] 数据集ID**（可以是一个或多个）|`-d 100` 或 `-d 100 101`|
|`-fpe FPE`|使用哪个 Fingerprint 提取器类，默认即可|`-fpe DatasetFingerprintExtractor`|
|`-npfp NPFP`|提取 Fingerprint 使用的进程数（默认8）|`-npfp 4`|
|`--verify_dataset_integrity`|验证数据完整性（推荐使用）|添加此参数即可|
|`--no_pp`|只做 fingerprint 和 planning，不做预处理|调试时使用|
|`--clean`|清除旧的 fingerprint，强制重新生成|修改了数据必须加上|
|`-pl PL`|使用的 Experiment Planner 类名（默认即可）|一般不用改|
|`-gpu_memory_target`|指定目标显存（单位GB），影响 patch 和 batch|例如：`-gpu_memory_target 10`|
|`-preprocessor_name`|使用自定义的预处理类（默认即可）|`-preprocessor_name DefaultPreprocessor`|
|`-overwrite_target_spacing`|自定义重采样 spacing（需三个数字）|`-overwrite_target_spacing 1.0 0.8 0.8`|
|`-overwrite_plans_name`|自定义 plans 名字，避免覆盖默认计划|推荐使用例如：`-overwrite_plans_name plan_10GB`|
|`-c C [C ...]`|选择要运行的模型配置|`-c 3d_fullres` 或 `-c 2d 3d_fullres`|
|`-np NP [NP ...]`|各配置的进程数，可设为一个或多个值|`-np 4` 或 `-np 4 8`（和 -c 对应）|
|`--verbose`|打印详细日志（会禁用进度条）|调试用|

  

## 3. 训练模型

### 核心命令格式：

```bash
nnUNetv2_train XXX CONFIG all --npz
```

- `XXX`：数据集 ID。

- `CONFIG`：模型配置，`3d_fullres`, `3d_lowres`, `2d`

- `all`：训练所有 5 折交叉验证（可指定折编号，如 `0`、`1` 等）。

  

### 参数解析：

`**nnUNetv2_train**`参数如下：

**必填参数：**

|参数名|说明|示例|
|---|---|---|
|`dataset_name_or_id`|任务ID或数据集名称|`100` 或 `Dataset100_MYTASK`|
|`configuration`|模型配置|`3d_fullres`, `3d_lowres`, `2d`|
|`fold`|交叉验证fold（0-4）或 `all`|`0`, `1`, ... 或 `all`（训练5折）|

- `3d_fullres`：3D 全分辨率训练

- `3d_lowres`：3D 低分辨率级联训练

- `2d`：2D训练

  

**可选参数：**

|参数|说明|推荐使用情况|
|---|---|---|
|`-tr`|指定自定义Trainer类（默认是 `nnUNetTrainer`）|如有自定义训练逻辑才用|
|`-p`|使用自定义 plans 文件名|你在 `plan_and_preprocess` 使用 `-overwrite_plans_name` 时要一致|
|`-pretrained_weights`|用已有模型权重做 finetune 或 warm start|`-pretrained_weights /path/to/checkpoint.pth`|
|`-num_gpus`|指定训练时使用几个GPU|如有多GPU，设为 `2`，否则默认1|
|`--npz`|保存 softmax 结果为 `.npz` 文件（用于 ensemble）|建议打开|
|`--c`|继续训练上次 checkpoint|`--c` 断点续训|
|`--val`|只运行验证，不训练|`--val`|
|`--val_best`|用 `checkpoint_best` 做验证（默认是 final）|常用于精度分析|
|`--disable_checkpointing`|不保存模型checkpoint（调试用）|非常规情况使用|
|`-device`|设置计算设备：`cuda`, `cpu`, `mps`|通常不设，用 `CUDA_VISIBLE_DEVICES=X` 控制GPU编号|

  

**训练 3D 全分辨率模型**：

```bash
nnUNetv2_train XXX 3d_fullres all --npz
```

- `XXX`：数据集 ID。

- `3d_fullres`：3D 全分辨率模型。

- `all`：训练所有 5 折交叉验证（可指定折编号，如 `0`、`1` 等）。

  

**训练 2D 模型**（适合显存较小的设备）：

```bash
nnUNetv2_train XXX 2d all --npz
```

  

## 4. 模型预测

### 核心命令格式：

基于指定模型目录下的 checkpoint 对输入图像进行分割预测，命令如下：

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET -c CONFIG -f all
```

- **CONFIG**：模型配置，`3d_fullres`, `3d_lowres`, `2d`

### 参数解析：

`**nnUNetv2_predict**`参数如下：

|参数|必须|说明|示例|
|---|---|---|---|
|`-i`|✅|输入图像目录（支持多图像，需有 `_0000.nii.gz` 命名）|`-i ./imagesTs`|
|`-o`|✅|预测输出保存目录|`-o ./predictions`|
|`-d`|✅|数据集ID或名称（与训练时一致）|`-d 100` 或 `-d Dataset100_MYTASK`|
|`-c`|✅|配置名称，如 `3d_fullres`、`2d`|`-c 3d_fullres`|
|`-f`|✅|指定使用哪些fold的模型（可多个）|`-f 0 1 2 3 4`|
|`-p`|❌|使用的 plans 名称（如训练时用了 `-p plan_xxx`）|`-p plan_10GB`|
|`-tr`|❌|指定 trainer 类名（如果训练时用了自定义）|`-tr nnUNetTrainer`|
|`-chk`|❌|指定使用哪个 checkpoint，默认是 `checkpoint_final.pth`|`-chk checkpoint_best.pth`|
|`--disable_tta`|❌|禁用测试时数据增强（TTA），**加速推理**但会略降精度|建议测试阶段使用，加速|
|`--save_probabilities`|❌|保存 softmax 概率（用于后处理或 ensemble）|`--save_probabilities`|
|`-step_size`|❌|滑窗步长，默认 0.5，越大越快但精度可能下降|`-step_size 0.75`|
|`-device`|❌|指定运行设备：`cuda`, `cpu`, `mps`（不要用于 GPU 选择！）|默认不设|
|`CUDA_VISIBLE_DEVICES=X`|❌|设置使用的 GPU 编号（不是 `-device`）|`CUDA_VISIBLE_DEVICES=0`|
|`--continue_prediction`|❌|如果中断了预测，加上此参数可跳过已完成的预测||
|`-num_parts` `-part_id`|❌|分布式预测（大数据集并行推理用）||
|`-npp`, `-nps`|❌|控制预处理和保存结果时的进程数|3|
|`--disable_progress_bar`|❌|不显示进度条（适合服务器后台运行）||
|`-prev_stage_predictions`|❌|如果是 cascade 模型，需指定前阶段预测目录||

### 输入图像命名要求：

推理图像必须为 **单通道 NIfTI 文件，且命名为** `**xxx_0000.nii.gz**`，否则无法识别。

|正确格式|错误格式|
|---|---|
|`case01_0000.nii.gz`|`case01.nii.gz` ❌|
|`liver001_0000.nii.gz`|`liver001.nii` ❌|

  

### 常见错误：

|错误|可能原因|解决办法|
|---|---|---|
|`Invalid input file format`|文件不是 `_0000.nii.gz` 格式|改名为 `xxx_0000.nii.gz`|
|`No checkpoint found`|没有 `checkpoint_final.pth`|检查模型训练是否完成或设 `-chk`|
|`plans not found`|训练时用了 `-p plan_xxx`，推理没指定|加 `-p plan_xxx`|

### 推荐参数：

|目的|推荐参数|
|---|---|
|**最快速度**|`--disable_tta -step_size 0.75`|
|**最高精度**|默认参数|
|**节省磁盘空间**|不加 `--save_probabilities`|
|**集群并行推理**|用 `-num_parts`, `-part_id`, `CUDA_VISIBLE_DEVICES`|

  

## nnUNet训练自动化脚本：

```bash
#!/bin/bash

# 配置
DATASET_ID="001"
DATASET_NAME="ASPECTS"
MODEL="3d_fullres"
FOLD="all"
gpu_id=1

# 激活环境
conda activate nnunet

# Step 1: 验证并预处理
echo "===== Step 1: 数据验证与预处理 ====="
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity
if [ $? -ne 0 ]; then
    echo "预处理失败！"
    exit 1
fi

# Step 2: 训练
echo "===== Step 2: 训练 $MODEL 模型 ====="
CUDA_VISIBLE_DEVICES=$gpu_id nnUNetv2_train $DATASET_ID $MODEL $FOLD --npz
if [ $? -ne 0 ]; then
    echo "训练失败！"
    exit 1
fi

# Step 3: 推理（可选）
echo "===== Step 3: 推理 ====="
CUDA_VISIBLE_DEVICES=$gpu_id nnUNetv2_predict -i ${nnUNet_raw}/Dataset${DATASET_ID}_${DATASET_NAME}/imagesTs -o ${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/inference -d $DATASET_ID -c $MODEL -f all
if [ $? -ne 0 ]; then
    echo "推理失败！"
    exit 1
fi

echo "===== 训练完成！====="
```