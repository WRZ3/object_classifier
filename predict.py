"""
推理脚本：对单张图片、单个 LabelMe JSON 标注文件、或整个目录进行分类预测。

用法：
  # 直接对图片分类
  python predict.py image.jpg

  # 对 LabelMe JSON 里的每个标注框分类
  python predict.py annotation.json

  # 对整个目录批量分类（结构同 rawdataset：<session>/rgb/*.png + *.json）
  # 默认只打印“预测结果和 JSON 标签不一致”的项 + 最后的汇总，不逐行刷屏
  python predict.py the_data_need_to_be_testes/

  # 不传参数时，默认扫描 the_data_need_to_be_testes/
  python predict.py

  # 想看每一个标注框的逐行结果（不只是错误项），加 --verbose
  python predict.py the_data_need_to_be_testes/ --verbose
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

_ROOT      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_ROOT, "classifier_best.pth")
CLASSES_PATH = os.path.join(_ROOT, "classes.json")
DEFAULT_TEST_DIR = os.path.join(_ROOT, "the_data_need_to_be_testes")
IMG_SIZE   = 224

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(model_path, classes):
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def predict_crop(model, classes, crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = val_tf(Image.fromarray(crop_rgb)).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = logits.softmax(dim=1).squeeze(0).numpy()
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx])


def crop_polygon(image_bgr, points):
    pts = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return image_bgr[max(0, y):y+h, max(0, x):x+w]


def classify_json(model, classes, json_path):
    """对单个 LabelMe JSON 里的每个标注框分类，不打印。
    返回 (img_name, results)；results 是 [{label, pred, conf, match}, ...]。
    读图失败时返回 (None, 错误原因字符串)。
    """
    with open(json_path) as f:
        data = json.load(f)
    img_name = data.get("imagePath", "")
    img_path = os.path.join(os.path.dirname(json_path), img_name)
    if not os.path.exists(img_path):
        return None, f"找不到对应图片: {img_path}"
    image = cv2.imread(img_path)
    if image is None:
        return None, f"无法读取图片: {img_path}"

    results = []
    for shape in data.get("shapes", []):
        label = shape["label"].strip()
        crop = crop_polygon(image, shape["points"])
        if crop.size == 0:
            continue
        pred, conf = predict_crop(model, classes, crop)
        results.append({"label": label, "pred": pred, "conf": conf, "match": pred == label})
    return img_name, results


def predict_json(model, classes, json_path):
    """单文件模式：对一个 LabelMe JSON 逐行打印每个标注框的结果。"""
    img_name, results = classify_json(model, classes, json_path)
    if img_name is None:
        print(f"  跳过（{results}）")
        return
    print(f"图片: {img_name}")
    for r in results:
        match = "✓" if r["match"] else "✗"
        print(f"  {match} JSON标签: {r['label']:<45}  预测: {r['pred']} ({r['conf']:.1%})")


def predict_dir(model, classes, root_dir, verbose=False):
    """扫描 <root_dir>/<session>/rgb/*.json（结构同 rawdataset），批量分类。
    默认只打印不一致的项，最后给出汇总（session / json 文件 / 图片），方便在几千条数据里定位问题。
    verbose=True 时逐行打印每个标注框的结果（包括一致的）。
    """
    sessions = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d, "rgb"))
    )
    if not sessions:
        sys.exit(f"{root_dir} 下没有找到任何 <session>/rgb/ 结构的数据")

    total_images, total_boxes = 0, 0
    errors = []

    for session in sessions:
        rgb_dir = os.path.join(root_dir, session, "rgb")
        json_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".json"))
        if not json_files:
            continue
        if verbose:
            print(f"\n=== session: {session} ===")

        for jf in json_files:
            img_name, results = classify_json(model, classes, os.path.join(rgb_dir, jf))
            if img_name is None:
                errors.append({"session": session, "file": jf, "reason": results})
                continue

            total_images += 1
            if verbose:
                print(f"图片: {img_name}")
            for r in results:
                total_boxes += 1
                if verbose:
                    match = "✓" if r["match"] else "✗"
                    print(f"  {match} JSON标签: {r['label']:<45}  预测: {r['pred']} ({r['conf']:.1%})")
                if not r["match"]:
                    errors.append({
                        "session": session, "file": jf, "image": img_name,
                        "label": r["label"], "pred": r["pred"], "conf": r["conf"],
                    })

    print(f"\n共处理 {total_images} 张图片，{total_boxes} 个标注框")
    if not errors:
        print("全部预测结果与 JSON 标签一致，没有发现问题。")
        return

    print(f"发现 {len(errors)} 处需要关注：\n")
    for e in errors:
        if "reason" in e:
            print(f"  [{e['session']}] {e['file']}  —— 跳过：{e['reason']}")
        else:
            print(f"  [{e['session']}] {e['file']}  图片: {e['image']}  "
                  f"JSON标签: {e['label']:<30}  预测: {e['pred']} ({e['conf']:.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", nargs="?", default=DEFAULT_TEST_DIR,
        help="图片路径（.jpg/.png）、LabelMe JSON 路径，或整个目录（结构同 rawdataset）。"
             "不传时默认使用 the_data_need_to_be_testes/",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="目录批量模式下逐行打印每个标注框的结果（默认只打印不一致的项 + 最后汇总）",
    )
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"模型文件不存在: {MODEL_PATH}\n请先运行 python train.py")

    with open(CLASSES_PATH) as f:
        classes = json.load(f)

    model = load_model(MODEL_PATH, classes)
    print(f"模型加载完成，共 {len(classes)} 个类别\n")

    path = args.input
    if not os.path.exists(path):
        sys.exit(f"路径不存在: {path}")

    if os.path.isdir(path):
        predict_dir(model, classes, path, verbose=args.verbose)
    elif path.endswith(".json"):
        predict_json(model, classes, path)
    else:
        image = cv2.imread(path)
        if image is None:
            sys.exit(f"无法读取图片: {path}")
        pred, conf = predict_crop(model, classes, image)
        print(f"预测结果: {pred}  置信度: {conf:.1%}")


if __name__ == "__main__":
    main()
