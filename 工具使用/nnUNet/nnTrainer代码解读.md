## 初始化阶段`__init__`

### 1.1 **设备与分布式配置**

```python
self.is_ddp = dist.is_available() and dist.is_initialized()
self.local_rank = 0 if not self.is_ddp else dist.get_rank()
self.device = device

if self.is_ddp:
    self.device = torch.device(type='cuda', index=self.local_rank)
else:
    if self.device.type == 'cuda':
        self.device = torch.device(type='cuda', index=0)
```

**说明**：自动检测分布式训练环境，为每个GPU分配正确的设备索引。

### 1.2 **路径与配置管理**

```python
self.plans_manager = PlansManager(plans)
self.configuration_manager = self.plans_manager.get_configuration(configuration)
self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                               self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration)
self.output_folder = join(self.output_folder_base, f'fold_{fold}')
```

**说明**：根据plans和configuration自动构建标准化的输出目录结构。

### 1.3 **核心超参数设置**

```python
self.initial_lr = 1e-2              # 初始学习率
self.weight_decay = 3e-5            # 权重衰减
self.oversample_foreground_percent = 0.33  # 前景过采样比例
self.num_iterations_per_epoch = 250        # 每epoch训练迭代数
self.num_val_iterations_per_epoch = 50     # 每epoch验证迭代数
self.num_epochs = 1000                      # 总epoch数
self.enable_deep_supervision = True        # 启用深度监督
```

### 1.4 **标签管理与损失函数准备**

```python
self.label_manager = self.plans_manager.get_label_manager(dataset_json)
# 支持常规分类训练和区域(Region)训练
```

## 训练阶段

```python
def run_training(self):
    self.on_train_start()

    for epoch in range(self.current_epoch, self.num_epochs):
        self.on_epoch_start()

        self.on_train_epoch_start()
        train_outputs = []
        for batch_id in range(self.num_iterations_per_epoch):
            train_outputs.append(self.train_step(next(self.dataloader_train)))
        self.on_train_epoch_end(train_outputs)

        with torch.no_grad():
            self.on_validation_epoch_start()
            val_outputs = []
            for batch_id in range(self.num_val_iterations_per_epoch):
                val_outputs.append(self.validation_step(next(self.dataloader_val)))
            self.on_validation_epoch_end(val_outputs)

        self.on_epoch_end()

    self.on_train_end()
```