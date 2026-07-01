# Object Classifier

用 EfficientNet-B0 对厨具/保温瓶等物体进行细粒度分类，基于 LabelMe polygon 标注数据集训练。

## 支持的类别（12类）

| 类别 | 说明 |
|------|------|
| `blue_pan` | 蓝色炒锅 |
| `red_pan` | 红色炒锅 |
| `dark_red_pan` | 深红色炒锅 |
| `pan_package` | 炒锅（纸箱包装） |
| `vacuum_bottle_golden_lid` | 保温壶·金色盖 |
| `vacuum_bottle_package_golden_lid` | 保温壶·金色盖（包装盒） |
| `vacuum_bottle_white_lid` | 保温壶·白色盖 |
| `vacuum_bottle_package_white_lid` | 保温壶·白色盖（包装盒） |
| `vacuum_bottle_pinkish_yellow` | 保温壶·粉黄色 |
| `vacuum_bottle_package_pinkish_yellow` | 保温壶·粉黄色（包装盒） |
| `vacuum_bottle_medium_blue` | 保温壶·中蓝色 |
| `vacuum_bottle_package_medium_blue` | 保温壶·中蓝色（包装盒） |

## 快速开始

### 0. 准备虚拟环境

建议用独立的 conda/venv 环境，避免和系统 python 或其他项目的依赖冲突：

```bash
conda create -n object-classifier python=3.10 -y
conda activate object-classifier
```



### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

将 LabelMe 标注的数据放入 `rawdataset/`，结构如下：

```
rawdataset/
└── <session_name>/
    └── rgb/
        ├── image_001.png
        ├── image_001.json   ← LabelMe polygon 标注
        ├── image_002.png
        └── image_002.json
```

然后运行数据准备脚本，自动裁剪 polygon 区域并按 80/20 分 train/val：

```bash
python prepare_dataset.py
```

### 3. 训练

```bash
python train.py
```

训练完成后模型保存至 `classifier_best.pth`，类别列表保存至 `classes.json`。

### 4. 推理（使用预训练模型）

仓库已附带训练好的模型权重 `classifier_best.pth`（验证集准确率 100%），可以跳过步骤 2–3 直接推理：

### 推理命令

```bash
# 对单张图片分类
python predict.py path/to/image.jpg

# 对 LabelMe JSON 里的每个标注框分类（同时对比 JSON 标签）
python predict.py path/to/annotation.json

# 批量分类：把待测数据按 rawdataset 同样的结构放进 the_data_need_to_be_testes/
#   the_data_need_to_be_testes/<session_name>/rgb/image_001.png + image_001.json
# 不传参数时默认扫描这个目录。
# 默认只打印“预测结果和 JSON 标签不一致”的项 + 最后的汇总（session/文件/图片名），
# 数据量大时不用逐行翻，方便定位问题。
python predict.py
python predict.py the_data_need_to_be_testes/

# 想看每个标注框的完整逐行结果（包括预测正确的），加 --verbose
python predict.py the_data_need_to_be_testes/ --verbose
```

## 项目结构

```
object-classifier/
├── rawdataset/                    # 训练用原始数据（自备，放 LabelMe 标注，见"2. 准备数据集"）
├── classifier_dataset/            # prepare_dataset.py 自动生成，裁剪后的 train/val 图片，不用手动放
├── the_data_need_to_be_testes/    # 待推理/测试数据（自备，结构同 rawdataset，见"批量分类"）
├── prepare_dataset.py      # 裁剪 rawdataset → classifier_dataset，80/20 分割
├── train.py                # EfficientNet-B0 fine-tune 训练脚本
├── predict.py              # 推理脚本（支持单图 / LabelMe JSON / 目录批量）
├── classifier_best.pth     # 预训练模型权重（可直接用于推理）
├── classes.json            # 类别列表
├── requirements.txt
└── .gitignore
```

## 训练配置

| 参数 | 值 |
|------|----|
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 30 |
| Batch size | 32 |
| Input size | 224×224 |
| 数据增强 | 水平翻转、颜色抖动、随机旋转±15° |
