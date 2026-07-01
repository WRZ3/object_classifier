"""
用 EfficientNet-B0 fine-tune 物体分类器。
训练完成后模型保存到 classifier_best.pth。
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

_ROOT       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(_ROOT, "classifier_dataset")
SAVE_PATH   = os.path.join(_ROOT, "classifier_best.pth")
EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 1e-4
IMG_SIZE    = 224

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class NonEmptyImageFolder(datasets.ImageFolder):
    """跳过没有图片的类文件夹，兼容数据还没补全的情况。"""
    def find_classes(self, directory):
        classes = sorted(
            d.name for d in os.scandir(directory)
            if d.is_dir() and any(
                f.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))
                for f in os.scandir(d.path)
            )
        )
        if not classes:
            raise FileNotFoundError(f"{directory} 下没有找到任何图片")
        return classes, {c: i for i, c in enumerate(classes)}

train_ds = NonEmptyImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_tf)
val_ds   = NonEmptyImageFolder(os.path.join(DATASET_DIR, "val"),   transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

num_classes = len(train_ds.classes)
print(f"类别数: {num_classes} → {train_ds.classes}")
print(f"训练集: {len(train_ds)}  验证集: {len(val_ds)}")

# 保存类别映射，推理时用
with open(os.path.join(_ROOT, "classes.json"), "w") as f:
    json.dump(train_ds.classes, f, ensure_ascii=False, indent=2)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    scheduler.step()

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            val_correct += (model(imgs).argmax(1) == labels).sum().item()
            val_total += imgs.size(0)

    train_acc = correct / total
    val_acc   = val_correct / val_total if val_total else 0.0
    print(f"Epoch {epoch:02d}/{EPOCHS}  loss={total_loss/total:.4f}  "
          f"train={train_acc:.2%}  val={val_acc:.2%}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"model": model.state_dict(), "classes": train_ds.classes}, SAVE_PATH)
        print(f"  ✓ 保存最优模型 (val={best_acc:.2%})")

print(f"\n训练完成，最优验证集准确率: {best_acc:.2%}")
print(f"模型已保存至: {SAVE_PATH}")
