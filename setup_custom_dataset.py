"""
setup_custom_dataset.py
=======================
Script tự động:
1. Tạo corre_hico.json từ file JSON annotation
2. Kiểm tra cấu trúc thư mục
3. In ra các thông số cần dùng khi chạy train

Cách dùng:
    python setup_custom_dataset.py \
        --train_json path/to/trainval_hico.json \
        --test_json  path/to/test_hico.json \
        --output_dir data/custom_dataset/annotations
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter


def analyze_dataset(json_path: str):
    with open(json_path) as f:
        data = json.load(f)

    obj_cats = set()
    verb_cats = set()
    subject_cats = Counter()
    object_cats = Counter()

    for item in data:
        ann_map = {ann['id']: ann for ann in item['annotations']}
        for ann in item['annotations']:
            obj_cats.add(ann['category_id'])
        for hoi in item['hoi_annotation']:
            verb_cats.add(hoi['category_id'])
            sid = hoi['subject_id']
            oid = hoi['object_id']
            if sid in ann_map:
                subject_cats[ann_map[sid]['category_id']] += 1
            if oid in ann_map:
                object_cats[ann_map[oid]['category_id']] += 1

    return {
        'num_images': len(data),
        'obj_cats': sorted(obj_cats),
        'max_obj_cat': max(obj_cats),
        'verb_cats': sorted(verb_cats),
        'num_verb_classes': len(verb_cats),
        'subject_cats': dict(subject_cats),
        'object_cats': dict(object_cats),
        'data': data,
    }


def build_corre_matrix(data, num_obj: int, num_verb: int) -> list:
    """Tạo ma trận tương quan HOI [num_obj x num_verb]"""
    corre = [[0] * num_verb for _ in range(num_obj)]
    for item in data:
        ann_map = {ann['id']: ann for ann in item['annotations']}
        for hoi in item['hoi_annotation']:
            oid = hoi['object_id']
            vid = hoi['category_id']
            if oid in ann_map:
                obj_cat = ann_map[oid]['category_id']  # 1-indexed
                if 1 <= obj_cat <= num_obj and 1 <= vid <= num_verb:
                    corre[obj_cat - 1][vid - 1] = 1
    return corre


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', required=True)
    parser.add_argument('--test_json',  required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("Analyzing TRAIN split...")
    train_info = analyze_dataset(args.train_json)
    print(f"  Images      : {train_info['num_images']}")
    print(f"  Object cats : {train_info['obj_cats']}")
    print(f"  Verb cats   : {train_info['verb_cats']}")
    print(f"  Subject cats: {train_info['subject_cats']}")
    print(f"  Object cats : {train_info['object_cats']}")

    print()
    print("Analyzing TEST/VAL split...")
    test_info = analyze_dataset(args.test_json)
    print(f"  Images      : {test_info['num_images']}")

    # Merge all data để build corre matrix
    all_data = train_info['data'] + test_info['data']
    num_obj  = max(train_info['max_obj_cat'], test_info['max_obj_cat'])
    num_verb = max(train_info['num_verb_classes'], test_info['num_verb_classes'])

    print()
    print(f"Building corre matrix [{num_obj} x {num_verb}]...")
    corre = build_corre_matrix(all_data, num_obj, num_verb)
    corre_path = output_dir / 'corre_hico.json'
    with open(corre_path, 'w') as f:
        json.dump(corre, f)
    print(f"  Saved -> {corre_path}")

    # Copy annotation files
    import shutil
    train_dst = output_dir / 'trainval_hico.json'
    test_dst  = output_dir / 'test_hico.json'
    shutil.copy(args.train_json, train_dst)
    shutil.copy(args.test_json,  test_dst)
    print(f"  Copied trainval_hico.json -> {train_dst}")
    print(f"  Copied test_hico.json     -> {test_dst}")

    print()
    print("=" * 55)
    print("✅ DONE! Dùng các tham số sau khi chạy train:")
    print()
    print(f"  --num_obj_classes  {num_obj}")
    print(f"  --num_verb_classes {num_verb}")
    print()
    print("Cấu trúc thư mục cần có:")
    print("""
  data/custom_dataset/
  ├── annotations/
  │   ├── trainval_hico.json   ← train split
  │   ├── test_hico.json       ← val/test split
  │   └── corre_hico.json      ← vừa tạo
  └── images/
      ├── train/               ← ảnh train
      └── test/                ← ảnh val/test
    """)
    print("=" * 55)


if __name__ == '__main__':
    main()
