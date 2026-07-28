---
Date: 2026-07-27
tags:
  - nVidia驱动
---

![[Nvidia驱动安装-1785144182887.webp]]
![[Nvidia驱动安装-1785144193863.webp]]

## 问题根源分析

- **权限错误**：`Operation not permitted` 通常意味着该文件被系统或进程锁定（例如被 `apparmor`、`selinux` 保护，或者有不可变的属性），或者之前安装的驱动残留文件有特殊权限。
- **安装未完成**：你运行 `sudo ./NVIDIA-Linux-x86_64-595.71.05.run` 后，只显示了 `Uncompressing...` 就退出了，没有显示后续的安装交互界面。说明安装程序可能检测到冲突（如 X Server 正在运行）或依赖缺失，自动终止了。
- **库文件缺失**：`libnvidia-ml.so` 是 `nvidia-smi` 依赖的库，未找到说明驱动根本没有被安装到系统目录（如 `/usr/lib`）。

## 解决方法：

### Step1：彻底清除旧驱动残留

首先，需要清理可能存在的旧 NVIDIA 驱动文件和配置，避免干扰新安装。

```sh
# 停止所有可能使用GPU的进程（如果有）
sudo pkill -f nvidia

# 如果系统有显示管理器（如gdm, lightdm），先停止它（如果是远程SSH，通常不需要）
sudo systemctl stop gdm   # 或 lightdm，取决于你的环境

# 使用官方卸载脚本（如果存在）
sudo /usr/bin/nvidia-uninstall   # 若不存在则跳过

# 强制删除残留的NVIDIA相关文件和目录
sudo rm -rf /usr/lib/nvidia*
sudo rm -rf /usr/lib/xorg/modules/drivers/nvidia*
sudo rm -rf /usr/bin/nvidia-*
sudo rm -rf /etc/ld.so.conf.d/nvidia*
sudo rm -rf /var/lib/nvidia*
sudo rm -rf /usr/src/nvidia-*
```

**注意**：如果某些文件仍提示 `Operation not permitted`，检查是否有 `chattr` 设置了不可变属性：

```sh
lsattr /usr/bin/nvidia-bug-report.sh   # 查看属性
sudo chattr -i /usr/bin/nvidia-bug-report.sh   # 去掉不可变属性
sudo rm /usr/bin/nvidia-bug-report.sh
```

### Step2：安装必要的编译依赖

安装 NVIDIA 驱动需要内核头文件和编译工具：

```sh
sudo apt update
sudo apt install build-essential dkms linux-headers-$(uname -r) -y
```

### Step3: 重新安装驱动（带推荐参数）

建议使用以下命令重新执行安装，并添加一些参数来跳过可能导致中断的检查：

```sh
sudo ./NVIDIA-Linux-x86_64-595.71.05.run \
    --no-x-check \
    --no-cc-version-check \
    --dkms \
    --silent
```
如果希望看到安装过程以便排查问题，可以去掉 `--silent`。  
安装过程中，如果提示“是否注册 DKMS”，选择 **Yes**。  
如果提示“是否更新 X 配置文件”，选择 **No**（除非你知道需要）。

### Step4：安装完成后重新加载内核模块
```sh
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_drm
```
然后验证：
```sh
nvidia-smi
```
如果仍然报错找不到库，可以尝试刷新动态链接库缓存：
```sh
sudo ldconfig
```