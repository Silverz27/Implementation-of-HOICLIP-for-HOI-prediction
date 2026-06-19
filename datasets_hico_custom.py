# datasets/hico.py - Modified for custom dataset
# Thay thế file gốc datasets/hico.py trong HOICLIP repo

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image

import datasets.transforms as T


class HICODetection(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries):
        self.img_set = img_set
        self.img_folder = img_folder

        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        self.annotations = annotations
        self._transforms = transforms
        self.num_queries = num_queries

        # ── Custom class definitions ──────────────────────────────────
        # Thay đổi các list này theo dataset của bạn
        # object_classes: tên các object (theo thứ tự category_id, 1-indexed)
        self._valid_obj_ids = list(range(1, 4))  # category_id 1, 2, 3

        self._valid_verb_ids = list(range(1, 9))  # verb category_id 1..8

        # Tên object tương ứng category_id (dùng cho CLIP text embedding)
        self.object_classes = [
            'vehicle',   # category_id 1
            'unknown',   # category_id 2 (không có trong data)
            'person',    # category_id 3 (subject)
        ]

        # Tên verb/action tương ứng category_id (dùng cho CLIP text embedding)
        self.verb_classes = [
            'action_1',  # verb category_id 1 → đổi tên thực tế của bạn
            'action_2',  # verb category_id 2
            'action_3',  # verb category_id 3
            'action_4',  # verb category_id 4
            'action_5',  # verb category_id 5
            'action_6',  # verb category_id 6
            'action_7',  # verb category_id 7
            'action_8',  # verb category_id 8
        ]
        # ─────────────────────────────────────────────────────────────

        self.num_obj_classes = len(self._valid_obj_ids)   # 3
        self.num_verb_classes = len(self._valid_verb_ids) # 8

        self._correct_mat = None

    def load_correct_mat(self, path):
        self._correct_mat = np.load(path) if path.endswith('.npy') else json.load(open(path))
        self._correct_mat = np.array(self._correct_mat)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_anno = self.annotations[idx]

        img_path = os.path.join(self.img_folder, img_anno['file_name'])
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # ── Boxes ────────────────────────────────────────────────────
        boxes = []
        for ann in img_anno['annotations']:
            x1, y1, x2, y2 = ann['bbox']
            boxes.append([x1, y1, x2, y2])

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Object labels (0-indexed)
        obj_labels = torch.tensor(
            [ann['category_id'] - 1 for ann in img_anno['annotations']],
            dtype=torch.int64
        )

        # ── HOI annotations ──────────────────────────────────────────
        # subject_boxes, object_boxes, verb labels
        ann_id_to_idx = {ann['id']: i for i, ann in enumerate(img_anno['annotations'])}

        hois = img_anno['hoi_annotation']
        sub_boxes, obj_boxes = [], []
        verb_labels = []

        for hoi in hois:
            sid = hoi['subject_id']
            oid = hoi['object_id']
            if sid not in ann_id_to_idx or oid not in ann_id_to_idx:
                continue
            s_idx = ann_id_to_idx[sid]
            o_idx = ann_id_to_idx[oid]
            sub_boxes.append(boxes[s_idx])
            obj_boxes.append(boxes[o_idx])

            # Verb label: one-hot vector of size num_verb_classes
            verb_vec = [0] * self.num_verb_classes
            verb_idx = hoi['category_id'] - 1  # 0-indexed
            if 0 <= verb_idx < self.num_verb_classes:
                verb_vec[verb_idx] = 1
            verb_labels.append(verb_vec)

        if len(sub_boxes) == 0:
            # Fallback: dummy
            sub_boxes = torch.zeros((1, 4), dtype=torch.float32)
            obj_boxes = torch.zeros((1, 4), dtype=torch.float32)
            verb_labels = torch.zeros((1, self.num_verb_classes), dtype=torch.float32)
        else:
            sub_boxes = torch.stack(sub_boxes)
            obj_boxes = torch.stack(obj_boxes)
            verb_labels = torch.as_tensor(verb_labels, dtype=torch.float32)

        # ── Build target dict ─────────────────────────────────────────
        target = {}
        target['orig_size'] = torch.as_tensor([h, w])
        target['size'] = torch.as_tensor([h, w])
        target['boxes'] = boxes
        target['labels'] = obj_labels
        target['iscrowd'] = torch.zeros(len(boxes), dtype=torch.int64)
        target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target['id'] = img_anno['img_id']

        target['hoi_labels'] = verb_labels
        target['sub_boxes'] = sub_boxes
        target['obj_boxes'] = obj_boxes

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def make_hico_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'

    # Mapping tên split -> tên file
    PATHS = {
        'train': (root / 'images' / 'train', root / 'annotations' / 'trainval_hico.json'),
        'val':   (root / 'images' / 'test',  root / 'annotations' / 'test_hico.json'),
    }

    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.json'

    img_folder, anno_file = PATHS[image_set]

    dataset = HICODetection(
        image_set,
        img_folder,
        anno_file,
        transforms=make_hico_transforms(image_set),
        num_queries=args.num_queries
    )

    if CORRECT_MAT_PATH.exists():
        dataset.load_correct_mat(str(CORRECT_MAT_PATH))

    return dataset
