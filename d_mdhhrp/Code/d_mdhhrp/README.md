# d_mdhhrp

独立的动态多中心居家医疗路径规划实现。
当前版本聚焦于**纯路径规划 + 动态患者调度**，不包含中心选址或中心激活逻辑。

支持两类输入数据：

- Solomon 静态实例：`load_solomon_instance()`
- Solomon 动态混合实例：`load_solomon_dynamic_instance()`，会把一部分患者按比例转换为动态患者并生成到达时间
- 动态混合实例导出：`save_hybrid_instance_to_json()`，可配合 `load_hybrid_instance_from_json()` 完成训练/测试数据闭环

## 模块概览

- `models.py`：核心实体、路径和动态环境
- `data_loader.py`：混合实例构造和随机实例生成
- `solution_converter.py`：路径与解对象之间的转换
- `operators.py`：ALNS 破坏/修复算子
- `policy.py`：PPO/算子选择策略与奖励
- `simulator.py`：事件驱动仿真入口
- `runner.py`：轻量 demo 入口

## 使用方式

先通过 `generate_random_hybrid_instance()` 生成测试实例，再用 `build_environment()` 构造环境，最后用 `DynamicSchedulingSimulator` 运行一次。

如果使用 Solomon 实例：

- 纯静态测试：先通过 `load_solomon_instance()` 指定预设中心，再直接进入环境构造和仿真流程。
- 动态测试：先通过 `load_solomon_dynamic_instance()` 将一部分客户转换成动态患者，再进入环境构造和仿真流程。

如果你需要做训练集管理，可以先生成动态混合实例，再用 `save_hybrid_instance_to_json()` 导出为统一 JSON。

## 批量数据集生成

批量生成多个动态混合实例用于 GNN 模型训练。

### 使用示例

```python
from d_mdhhrp.runner import run_batch_dataset_generation

# 生成数据集：使用所有 Solomon 文件，生成 4 个动态比例 × 4 个策略 × 2 个 seed
metadata = run_batch_dataset_generation(
    solomon_dir='/Users/marc/Documents/Code/in',
    output_dir='/Users/marc/Documents/Code/dataset',
    num_files=None,  # None 表示使用所有文件
    verbose=True
)

print(f"Total instances: {metadata['total_instances']}")
print(f"Train: {metadata['train_instances']}, Val: {metadata['val_instances']}, Test: {metadata['test_instances']}")
```

### 输出目录结构

```
dataset/
  ├── train/
  │   ├── c101_000000.json
  │   ├── c101_000001.json
  │   └── ...
  ├── val/
  │   ├── ...
  ├── test/
  │   ├── ...
  └── metadata.json
```

### 参数配置

- `dynamic_ratios`: [0.2, 0.3, 0.4, 0.5] — 动态患者占比
- `dynamic_strategies`: ['early', 'late', 'uniform', 'midpoint'] — 患者到达时间生成策略
- `num_seeds_per_config`: 2 — 每个参数组合生成的随机种子数
- `num_centers_range`: (1, 4) — 中心数量范围

### 加载数据集

```python
from d_mdhhrp.dataset_batch_generator import load_dataset_split
from d_mdhhrp.data_loader import load_hybrid_instance_from_json

# 加载训练集
train_instances = load_dataset_split('/Users/marc/Documents/Code/dataset', split='train')

for file_path, metadata in train_instances[:3]:
    inst = load_hybrid_instance_from_json(file_path)
    print(f"{metadata['id']}: {len(inst.scheduled_patients)} scheduled, {len(inst.dynamic_patients)} dynamic")
```

### 数据集规模

使用所有 57 个 Solomon 文件、4 个动态比例、4 个策略、2 个 seed，会生成约 **1824+ 个实例**。
