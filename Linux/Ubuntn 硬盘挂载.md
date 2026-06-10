---
Date: 2026-06-10
tags:
  - 硬盘挂载
---
### 1. 识别新接入的硬盘

```sh
# 查看块设备列表，通常新接入的硬盘会显示为 sdb, sdc 等
lsblk
```
![](https://pic1.imgdb.cn/item/6a28d9a0edae85a628530233.png)
### 2. 挂载硬盘

在 Linux 中，需要创建一个目录作为“挂载点”，然后将硬盘挂载到该目录上。

```sh
# 1. 创建一个挂载点目录
sudo mkdir -p /data2

# 2. 挂载硬盘
sudo mount /dev/sdd2 /data2
```
![](https://pic1.imgdb.cn/item/6a28da0eedae85a628530243.png)

### 3. 拷贝数据至服务器（推荐 rsync）

假设您要将硬盘里的 `medical_data` 文件夹拷贝到服务器的 `/data0/` 目录下，使用 `rsync` 可以实现**断点续传**和**进度显示**：

```sh
# 使用 rsync 进行拷贝
rsync -avhP /data2/medical_data/ /data0/medical_data/
```

**参数说明（结合您的使用习惯）**：

- `-a`：归档模式，保留文件权限、时间戳、软链接等。
- `-v`：详细输出，显示正在传输的文件。
- `-h`：以人类可读的格式（如 MB, GB）输出文件大小。
- `-P`：等同于 `--partial --progress`。**`--partial` 保留传输中断的不完整文件，`--progress` 显示实时进度条**。这是实现**断点续传**的核心参数。如果传输中断，只需**再次运行完全相同的命令**，它会自动从断点处继续。
### 4. 安全卸载硬盘

数据拷贝完成并确认无误后，**千万不要直接拔出硬盘**，必须先卸载以防止数据损坏：
```sh
# 卸载硬盘
sudo umount /data2
```
