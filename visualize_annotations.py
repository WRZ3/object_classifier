"""
可视化 LabelMe JSON 标注：把 json 里每个标注框画到对应图片上（多边形轮廓 + 序号 + 标签文字），
并把每个标注框单独裁剪保存，方便肉眼核对某个标注到底标错了没有。

不依赖 torch，只用 opencv/numpy，跟 predict.py 是否装了 torch 无关。

用法：
  # 只传 json，图片路径按 imagePath 从同目录找
  python visualize_annotations.py path/to/xxx.json

  # 输出默认放在 json 同目录下的 <basename>_annotated.png 和 <basename>_crops/
  # 也可以自己指定输出目录
  python visualize_annotations.py path/to/xxx.json --out-dir /tmp/check

  # 只想看某个标签的框（比如只看 red_pan，忽略一堆 vacuum_bottle）
  python visualize_annotations.py path/to/xxx.json --label red_pan
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

# 给每个不同的 label 固定分配一个颜色（BGR），同一个 label 每次跑颜色一致，方便对照
_PALETTE = [
    (60, 60, 255), (255, 144, 30), (0, 200, 0), (0, 215, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (180, 105, 255), (0, 255, 128), (200, 200, 0), (100, 100, 255),
]


def color_for_label(label, all_labels):
    idx = sorted(set(all_labels)).index(label)
    return _PALETTE[idx % len(_PALETTE)]


def load_annotation(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    img_name = data.get("imagePath", "")
    img_path = os.path.join(os.path.dirname(json_path), img_name)
    if not os.path.exists(img_path):
        # imagePath 记录的文件名有时和实际同目录 png 对不上，退而用 json 同名 png 兜底
        fallback = os.path.splitext(json_path)[0] + ".png"
        if os.path.exists(fallback):
            img_path = fallback
        else:
            sys.exit(f"找不到对应图片: {img_path}（也没有同名 png: {fallback}）")
    image = cv2.imread(img_path)
    if image is None:
        sys.exit(f"无法读取图片: {img_path}")
    return data, image, img_path


def draw_annotated(image, shapes, only_label=None):
    vis = image.copy()
    all_labels = [s["label"].strip() for s in shapes]
    for i, shape in enumerate(shapes):
        label = shape["label"].strip()
        if only_label and label != only_label:
            continue
        pts = np.array(shape["points"], dtype=np.int32)
        color = color_for_label(label, all_labels)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)

        score = shape.get("score")
        tag = f"#{i} {label}" + (f" ({score:.2f})" if score is not None else "")
        text_y = max(15, y - 6)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x, text_y - th - 4), (x + tw + 4, text_y + 2), color, -1)
        cv2.putText(vis, tag, (x + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def save_crops(image, shapes, out_dir, only_label=None):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i, shape in enumerate(shapes):
        label = shape["label"].strip()
        if only_label and label != only_label:
            continue
        pts = np.array(shape["points"], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        crop = image[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            continue
        safe_label = label.replace("/", "_")
        out_path = os.path.join(out_dir, f"{i:02d}_{safe_label}.png")
        cv2.imwrite(out_path, crop)
        saved.append(out_path)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_path", help="LabelMe JSON 标注文件路径")
    parser.add_argument("--out-dir", default=None,
                         help="输出目录，默认用 json 同目录（生成 <basename>_annotated.png 和 <basename>_crops/）")
    parser.add_argument("--label", default=None,
                         help="只显示/裁剪某一个标签，比如 --label red_pan（默认显示全部标注框）")
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        sys.exit(f"文件不存在: {args.json_path}")

    data, image, img_path = load_annotation(args.json_path)
    shapes = data.get("shapes", [])
    if not shapes:
        sys.exit("这个 json 里没有任何标注框（shapes 为空）")

    base = os.path.splitext(os.path.basename(args.json_path))[0]
    out_dir = args.out_dir or os.path.dirname(args.json_path)
    os.makedirs(out_dir, exist_ok=True)

    annotated_path = os.path.join(out_dir, f"{base}_annotated.png")
    vis = draw_annotated(image, shapes, only_label=args.label)
    cv2.imwrite(annotated_path, vis)

    crops_dir = os.path.join(out_dir, f"{base}_crops")
    saved = save_crops(image, shapes, crops_dir, only_label=args.label)

    print(f"原图: {img_path}")
    print(f"标注总览图（画出全部框+序号+标签）: {annotated_path}")
    print(f"逐框裁剪图（{len(saved)} 张，文件名格式 序号_标签.png）: {crops_dir}/")
    print()
    print("各标注框：")
    all_labels = [s["label"].strip() for s in shapes]
    for i, s in enumerate(shapes):
        label = s["label"].strip()
        if args.label and label != args.label:
            continue
        pts = np.array(s["points"], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        print(f"  #{i:02d}  label={label:<35} bbox=({x},{y},{x+w},{y+h})  score={s.get('score')}")


if __name__ == "__main__":
    main()
