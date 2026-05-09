"""
FLIR-Aligned 数据预处理脚本
=============================
项目根目录: E:\ccf_c\
数据集目录: E:\ccf_c\data\flir_align\

输入结构:
  E:\ccf_c\data\flir_align\
  ├── coco_annotations\
  │   ├── train_new.json
  │   └── test_new.json
  ├── thermal\
  │   ├── train\  (FLIR_xxxxx.jpeg)
  │   └── test\   (FLIR_xxxxx.jpeg)
  └── visible\
      ├── train\  (FLIR_xxxxx.jpeg)
      └── test\   (FLIR_xxxxx.jpeg)

输出结构:
  E:\ccf_c\data\FLIR_yolo\
  ├── data.yaml
  ├── images\
  │   ├── visible\
  │   │   ├── train\  (约3716张)
  │   │   ├── val\    (约413张，从train切出10%)
  │   │   └── test\   (1013张，官方原版)
  │   └── thermal\
  │       ├── train\
  │       ├── val\
  │       └── test\
  └── labels\
      ├── train\  (YOLO格式 txt，RGB和IR共用)
      ├── val\
      └── test\

运行方法:
  python preprocess_flir.py
"""

import json
import os
import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
#  路径配置（已根据你的目录固定好，不需要修改）
# ============================================================

# 原始数据集根目录
SOURCE_DIR = Path(r"E:\ccf_c\data\flir_align")

# 标注文件
TRAIN_JSON = SOURCE_DIR / "coco_annotations" / "train_new.json"
TEST_JSON  = SOURCE_DIR / "coco_annotations" / "test_new.json"

# 图像源目录
VIS_TRAIN_DIR = SOURCE_DIR / "visible"  / "train"
VIS_TEST_DIR  = SOURCE_DIR / "visible"  / "test"
THM_TRAIN_DIR = SOURCE_DIR / "thermal"  / "train"
THM_TEST_DIR  = SOURCE_DIR / "thermal"  / "test"

# 输出目录
OUTPUT_DIR = Path(r"E:\ccf_c\data\FLIR_yolo")

# val 划分比例和随机种子
VAL_RATIO   = 0.1
RANDOM_SEED = 42

# Windows 下符号链接需要管理员权限，默认用复制
USE_SYMLINK = False


# ============================================================
#  工具函数（不需要修改）
# ============================================================

def coco_to_yolo(bbox, img_w, img_h):
    """
    COCO bbox: [x_左上, y_左上, 宽, 高]  单位: 像素
    YOLO bbox: [x_中心, y_中心, 宽, 高]  归一化到 [0, 1]
    """
    x, y, w, h = bbox
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_n = w  / img_w
    h_n = h  / img_h
    # 防止标注溢出图像边界
    x_c = max(0.0, min(1.0, x_c))
    y_c = max(0.0, min(1.0, y_c))
    w_n = max(0.0, min(1.0, w_n))
    h_n = max(0.0, min(1.0, h_n))
    return x_c, y_c, w_n, h_n


def load_coco(json_path):
    """
    加载 COCO 格式标注。
    返回:
      images_info : {img_id: {file_name, width, height, ...}}
      annotations : {img_id: [(cat_id, bbox), ...]}
      categories  : {cat_id: name}
    """
    print(f"    读取: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    categories  = {c["id"]: c["name"] for c in data["categories"]}
    images_info = {img["id"]: img for img in data["images"]}

    annotations = defaultdict(list)
    n_skip = 0
    for ann in data["annotations"]:
        # 跳过 ignore / iscrowd 的框
        if ann.get("ignore", 0) or ann.get("iscrowd", 0):
            n_skip += 1
            continue
        # 跳过零面积框
        if ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0:
            n_skip += 1
            continue
        annotations[ann["image_id"]].append(
            (ann["category_id"], ann["bbox"])
        )

    if n_skip:
        print(f"    跳过 {n_skip} 个无效标注 (ignore/iscrowd/零面积)")

    return images_info, dict(annotations), categories


def place_file(src: Path, dst: Path):
    """复制（或符号链接）一个文件到目标位置"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if USE_SYMLINK:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def write_yolo_label(label_path: Path, anns, img_w, img_h):
    """将一张图的标注写成 YOLO 格式 txt"""
    with open(label_path, "w") as f:
        for cat_id, bbox in anns:
            x_c, y_c, w, h = coco_to_yolo(bbox, img_w, img_h)
            f.write(f"{cat_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def process_split(split_name, image_ids,
                  images_info, annotations,
                  vis_src_dir, thm_src_dir):
    """
    处理一个数据集划分 (train / val / test)。
    - 将图像放到输出目录
    - 将标注转换成 YOLO 格式 txt
    返回: (成功数, 缺失数, 各类别标注计数)
    """
    out_vis = OUTPUT_DIR / "images" / "visible" / split_name
    out_thm = OUTPUT_DIR / "images" / "thermal" / split_name
    out_lbl = OUTPUT_DIR / "labels" / split_name
    for d in [out_vis, out_thm, out_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    ok, miss = 0, 0
    cat_cnt  = Counter()

    for img_id in image_ids:
        info  = images_info[img_id]
        fname = info["file_name"]          # 如 FLIR_00258.jpeg
        iw    = info["width"]              # 640
        ih    = info["height"]             # 512

        vis_src = vis_src_dir / fname
        thm_src = thm_src_dir / fname

        # 容错：尝试 .jpg 后缀
        if not vis_src.exists():
            alt = vis_src_dir / fname.replace(".jpeg", ".jpg")
            if alt.exists():
                vis_src = alt
        if not thm_src.exists():
            alt = thm_src_dir / fname.replace(".jpeg", ".jpg")
            if alt.exists():
                thm_src = alt

        # 两个模态的图像都必须存在
        if not vis_src.exists() or not thm_src.exists():
            miss += 1
            if miss <= 5:
                print(f"    [警告] 找不到图像: {fname}")
                print(f"           visible 路径: {vis_src}  存在={vis_src.exists()}")
                print(f"           thermal 路径: {thm_src}  存在={thm_src.exists()}")
            continue

        # 放置图像
        place_file(vis_src, out_vis / fname)
        place_file(thm_src, out_thm / fname)

        # 写标注 txt（RGB 和 IR 共用同一套标注）
        stem = Path(fname).stem          # FLIR_00258
        anns = annotations.get(img_id, [])
        if anns:
            write_yolo_label(out_lbl / f"{stem}.txt", anns, iw, ih)
            for cat_id, _ in anns:
                cat_cnt[cat_id] += 1
        else:
            # 没有目标的图像也要创建空 txt
            # YOLO 训练要求每张图必须有对应的 txt 文件
            (out_lbl / f"{stem}.txt").touch()

        ok += 1

    return ok, miss, cat_cnt


# ============================================================
#  主流程
# ============================================================

def main():
    print("=" * 55)
    print("  FLIR-Aligned 数据预处理")
    print("=" * 55)

    # ---- 检查所有输入路径 ----
    check_paths = {
        "训练标注 (train_new.json)": TRAIN_JSON,
        "测试标注 (test_new.json)" : TEST_JSON,
        "visible/train"            : VIS_TRAIN_DIR,
        "visible/test"             : VIS_TEST_DIR,
        "thermal/train"            : THM_TRAIN_DIR,
        "thermal/test"             : THM_TEST_DIR,
    }
    missing = {k: v for k, v in check_paths.items() if not v.exists()}
    if missing:
        print("\n❌ 以下路径不存在，请检查数据集是否正确解压：")
        for name, path in missing.items():
            print(f"   [{name}]  {path}")
        return

    print("\n✅ 所有输入路径检查通过")

    # ---- 清空旧输出目录 ----
    if OUTPUT_DIR.exists():
        print(f"\n⚠  输出目录已存在，清空后重建: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # ---- 1. 加载 COCO 标注 ----
    print("\n[1/4] 加载 COCO 标注...")
    train_imgs, train_anns, categories = load_coco(TRAIN_JSON)
    test_imgs,  test_anns,  _          = load_coco(TEST_JSON)

    n_tr_ann = sum(len(v) for v in train_anns.values())
    n_te_ann = sum(len(v) for v in test_anns.values())
    print(f"  训练集: {len(train_imgs)} 张图,  {n_tr_ann} 个标注框")
    print(f"  测试集: {len(test_imgs)}  张图,  {n_te_ann}  个标注框")
    print(f"  类别  : {categories}")

    # ---- 2. 划分 train / val ----
    print(f"\n[2/4] 划分 train / val  "
          f"(val={VAL_RATIO*100:.0f}%, seed={RANDOM_SEED})...")

    all_ids = list(train_imgs.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(all_ids)

    val_n     = int(len(all_ids) * VAL_RATIO)
    val_ids   = all_ids[:val_n]
    train_ids = all_ids[val_n:]
    test_ids  = list(test_imgs.keys())

    print(f"  train : {len(train_ids)} 张")
    print(f"  val   : {len(val_ids)}   张")
    print(f"  test  : {len(test_ids)}  张 (官方，不动)")

    # ---- 3. 转换并组织文件 ----
    print("\n[3/4] 格式转换 & 组织文件...")

    print(f"\n  → train ({len(train_ids)} 张)")
    ok, miss, cc = process_split(
        "train", train_ids,
        train_imgs, train_anns,
        VIS_TRAIN_DIR, THM_TRAIN_DIR
    )
    print(f"    成功: {ok}  |  缺失: {miss}")
    print(f"    标注: { {categories[k]: v for k, v in sorted(cc.items())} }")

    print(f"\n  → val ({len(val_ids)} 张，来自 train 目录)")
    ok, miss, cc = process_split(
        "val", val_ids,
        train_imgs, train_anns,
        VIS_TRAIN_DIR, THM_TRAIN_DIR    # val 图像来自 train 目录
    )
    print(f"    成功: {ok}  |  缺失: {miss}")
    print(f"    标注: { {categories[k]: v for k, v in sorted(cc.items())} }")

    print(f"\n  → test ({len(test_ids)} 张)")
    ok, miss, cc = process_split(
        "test", test_ids,
        test_imgs, test_anns,
        VIS_TEST_DIR, THM_TEST_DIR
    )
    print(f"    成功: {ok}  |  缺失: {miss}")
    print(f"    标注: { {categories[k]: v for k, v in sorted(cc.items())} }")

    # ---- 4. 生成 data.yaml ----
    print("\n[4/4] 生成 data.yaml...")

    # 输出路径统一用正斜杠，YOLO 框架在 Windows 上也能识别
    out_posix = OUTPUT_DIR.as_posix()

    yaml_content = f"""\
# FLIR-Aligned Dataset  —  YOLO 格式
# 生成脚本 : preprocess_flir.py
# 标注版本 : train_new / test_new  (3类，car / person / bicycle)
# val 划分 : seed={RANDOM_SEED}, ratio={VAL_RATIO}  (从 train 里切)

# 数据集根目录（绝对路径）
path: {out_posix}

# ---- 标准单流 YOLO 字段 ----
train: images/visible/train
val:   images/visible/val
test:  images/visible/test

# ---- 双流框架额外字段 ----
# （不同的双流代码可能用不同的字段名，按需修改）
visible_train:  images/visible/train
visible_val:    images/visible/val
visible_test:   images/visible/test
thermal_train:  images/thermal/train
thermal_val:    images/thermal/val
thermal_test:   images/thermal/test

# 类别
nc: {len(categories)}
names: {list(categories.values())}
# 类别说明:
#   0 = car      (汽车)
#   1 = person   (行人)
#   2 = bicycle  (自行车)

# 样本数量:
#   train : {len(train_ids)}
#   val   : {len(val_ids)}
#   test  : {len(test_ids)}
"""

    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"  → {yaml_path}")

    # ---- 完成 ----
    print("\n" + "=" * 55)
    print("✅  预处理完成！")
    print("=" * 55)
    print(f"""
输出目录: {OUTPUT_DIR}

  FLIR_yolo\\
  ├── data.yaml
  ├── images\\
  │   ├── visible\\
  │   │   ├── train\\   {len(train_ids)} 张 RGB
  │   │   ├── val\\     {len(val_ids)} 张 RGB
  │   │   └── test\\    {len(test_ids)} 张 RGB
  │   └── thermal\\
  │       ├── train\\   {len(train_ids)} 张 IR
  │       ├── val\\     {len(val_ids)} 张 IR
  │       └── test\\    {len(test_ids)} 张 IR
  └── labels\\
      ├── train\\       {len(train_ids)} 个 txt
      ├── val\\         {len(val_ids)} 个 txt
      └── test\\        {len(test_ids)} 个 txt

验证步骤:
  1. 打开 labels\\train\\ 里任意一个 txt 文件
     每行应该是: 类别ID x中心 y中心 宽 高
     例如:       0 0.623437 0.485547 0.107812 0.103516
     所有数值应在 0~1 之间

  2. 用图像查看器打开同名的 visible 和 thermal 图像
     例如: FLIR_00258.jpeg
     确认两张是同一场景的 RGB + 红外配对

  3. 如果一切正常，下一步就是配置双流 YOLO baseline 代码
     把 data.yaml 的路径告诉 baseline 的训练脚本即可
""")


if __name__ == "__main__":
    main()
