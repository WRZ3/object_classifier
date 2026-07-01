"""
从 rawdataset 的 LabelMe JSON 裁出 polygon 区域，按 80/20 分到 train/val。
之后新增图片只需放进 rawdataset 对应文件夹，重新跑这个脚本即可。
"""
import json
import os
import random
import shutil
from collections import defaultdict

import cv2
import numpy as np

_ROOT          = os.path.dirname(os.path.abspath(__file__))
RAWDATASET_DIR = os.path.join(_ROOT, "rawdataset")
OUT_DIR        = os.path.join(_ROOT, "classifier_dataset")
VAL_RATIO      = 0.2
SEED           = 42

CLASSES = {
    "blue_pan", "red_pan", "dark_red_pan", "pan_package",
    "vacuum_bottle_golden_lid", "vacuum_bottle_package_golden_lid",
    "vacuum_bottle_white_lid", "vacuum_bottle_package_white_lid",
    "vacuum_bottle_pinkish_yellow", "vacuum_bottle_package_pinkish_yellow",
    "vacuum_bottle_medium_blue", "vacuum_bottle_package_medium_blue",
}

# JSON 里的拼写错误 → 正确类别名
TYPO_FIX = {
    "vacuum_bottlr_golden_lid":          "vacuum_bottle_golden_lid",
    "vacuum_bottle_package_gloden_lid":  "vacuum_bottle_package_golden_lid",
}

random.seed(SEED)


def crop_polygon(image_bgr, points):
    pts = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    x, y = max(0, x), max(0, y)
    return image_bgr[y:y+h, x:x+w]


def collect_crops():
    """返回 {label: [(crop_bgr, src_path, idx), ...]}"""
    crops = defaultdict(list)

    for session in sorted(os.listdir(RAWDATASET_DIR)):
        rgb_dir = os.path.join(RAWDATASET_DIR, session, "rgb")
        if not os.path.isdir(rgb_dir):
            continue

        for jf in sorted(f for f in os.listdir(rgb_dir) if f.endswith(".json")):
            json_path = os.path.join(rgb_dir, jf)
            with open(json_path) as f:
                data = json.load(f)

            img_name = data.get("imagePath", jf.replace(".json", ".png"))
            img_path = os.path.join(rgb_dir, img_name)
            if not os.path.exists(img_path):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            for i, shape in enumerate(data.get("shapes", [])):
                label = shape["label"].strip()
                label = TYPO_FIX.get(label, label)
                if label not in CLASSES:
                    continue
                crop = crop_polygon(image, shape["points"])
                if crop.size == 0:
                    continue
                crops[label].append((crop, f"{session}/{jf}", i))

    return crops


def split_and_save(crops):
    stats = {}
    for label, items in crops.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * VAL_RATIO)) if len(items) > 1 else 0
        splits = {"val": items[:n_val], "train": items[n_val:]}

        for split, split_items in splits.items():
            out_dir = os.path.join(OUT_DIR, split, label)
            os.makedirs(out_dir, exist_ok=True)
            for idx, (crop, src, shape_idx) in enumerate(split_items):
                fname = f"{idx:04d}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])

        stats[label] = {"train": len(splits["train"]), "val": len(splits["val"])}

    return stats


def main():
    # 清空旧数据
    for split in ("train", "val"):
        split_dir = os.path.join(OUT_DIR, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        # 预建全部 12 个类的文件夹（无数据的也保留）
        for cls in CLASSES:
            os.makedirs(os.path.join(split_dir, cls))

    print("裁剪中...")
    crops = collect_crops()
    stats = split_and_save(crops)

    print("\n完成！各类数量：")
    print(f"  {'标签':<45} {'train':>6} {'val':>5}")
    print("  " + "-" * 58)
    for label, s in sorted(stats.items()):
        print(f"  {label:<45} {s['train']:>6} {s['val']:>5}")
    print(f"\n  总计 train: {sum(s['train'] for s in stats.values())}")
    print(f"  总计 val:   {sum(s['val']   for s in stats.values())}")


if __name__ == "__main__":
    main()
