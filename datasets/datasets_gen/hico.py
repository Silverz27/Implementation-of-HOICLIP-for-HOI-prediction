"""
HICO detection dataset - Custom version for vehicle boarding/alighting dataset
Thay thế toàn bộ file datasets/hico.py
"""
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import torch
import torch.utils.data
import clip

import datasets.transforms as T


# ── Custom text labels ────────────────────────────────────────────────────────
# Key: (verb_idx, obj_idx) - đều 0-indexed
# Value: text description cho CLIP
# verb_idx = category_id - 1, obj_idx = category_id - 1
custom_text_label = {
    (0, 0): 'a person gets in the car',       # len_xe,      verb=1, obj=1
    (1, 0): 'a person gets out of the car',   # xuong_xe,    verb=2, obj=1
    (2, 0): 'a person opens the car door',    # mo_cua_xe,   verb=3, obj=1
    (3, 0): 'a person closes the car door',   # dong_cua_xe, verb=4, obj=1
    (4, 0): 'a person opens the trunk',       # mo_cop_xe,   verb=5, obj=1
    (5, 0): 'a person closes the trunk',      # dong_cop_xe, verb=6, obj=1
    (6, 0): 'the car stops',                  # xe_dung,     verb=7, obj=1
    (7, 0): 'the car moves',                  # xe_di,       verb=8, obj=1
}
# ─────────────────────────────────────────────────────────────────────────────


class HICODetection(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, anno_file, clip_feats_folder, transforms, num_queries, args):
        self.img_set = img_set
        self.img_folder = img_folder
        self.clip_feates_folder = clip_feats_folder

        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms
        self.num_queries = num_queries

        # ── Custom class definitions ──────────────────────────────────────────
        # category_id trong JSON là 1-indexed
        self._valid_obj_ids   = (1, 2, 3)   # car=1, unknown=2, person=3
        self._valid_verb_ids  = list(range(1, 9))  # verbs 1..8

        self.text_label_dict  = custom_text_label
        self.text_label_ids   = list(self.text_label_dict.keys())
        self.unseen_index     = []  # không dùng zero-shot
        # ─────────────────────────────────────────────────────────────────────

        total_anno = 0

        if img_set == 'train':
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                new_img_anno = []
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or \
                       hoi['object_id'] >= len(img_anno['annotations']):
                        new_img_anno = []
                        break
                    new_img_anno.append(hoi)
                if len(new_img_anno) > 0:
                    self.ids.append(idx)
                    img_anno['hoi_annotation'] = new_img_anno
                total_anno += len(new_img_anno)
        else:
            self.ids = list(range(len(self.annotations)))

        print("{} contains {} images and {} annotations".format(img_set, len(self.ids), total_anno))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.clip_preprocess = clip.load(args.clip_model, device)

    def __len__(self):
        return len(self.ids)

    def _get_img_path(self, file_name):
        """Tìm ảnh trong subfolder theo tên action (bỏ số cuối)"""
        # file_name vd: dong_cop_xe_0.jpg -> subfolder: dong_cop_xe
        parts = file_name.replace('.jpg', '').replace('.png', '').split('_')
        # Bỏ phần số cuối, ghép lại thành tên subfolder
        subfolder = '_'.join(parts[:-1])
        return self.img_folder / subfolder / file_name

    def __getitem__(self, idx):
        img_anno = self.annotations[self.ids[idx]]

        img_path = self._get_img_path(img_anno['file_name'])
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.img_set == 'train':
            classes = [(i, self._valid_obj_ids.index(obj['category_id']))
                       for i, obj in enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id'])
                       for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size']      = torch.as_tensor([int(h), int(w)])

        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes   = boxes[keep]
            classes = classes[keep]

            target['boxes']   = boxes
            target['labels']  = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area']    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img_0, target_0 = self._transforms[0](img, target)
                img, target     = self._transforms[1](img_0, target_0)

            clip_inputs = self.clip_preprocess(img_0)
            target['clip_inputs'] = clip_inputs
            kept_box_indices = [label[0] for label in target['labels']]
            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            hoi_labels = []

            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or \
                   hoi['object_id'] not in kept_box_indices:
                    continue

                verb_idx = self._valid_verb_ids.index(hoi['category_id'])
                obj_idx  = target['labels'][kept_box_indices.index(hoi['object_id'])].item()
                verb_obj_pair = (verb_idx, obj_idx)

                if verb_obj_pair not in self.text_label_ids:
                    continue

                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    i = sub_obj_pairs.index(sub_obj_pair)
                    verb_labels[i][verb_idx] = 1
                    hoi_labels[i][self.text_label_ids.index(verb_obj_pair)] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])

                    verb_label = [0] * len(self._valid_verb_ids)
                    verb_label[verb_idx] = 1

                    hoi_label = [0] * len(self.text_label_ids)
                    hoi_label[self.text_label_ids.index(verb_obj_pair)] = 1

                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]

                    verb_labels.append(verb_label)
                    hoi_labels.append(hoi_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)

            target['filename'] = img_anno['file_name']

            if len(sub_obj_pairs) == 0:
                target['obj_labels']  = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['hoi_labels']  = torch.zeros((0, len(self.text_label_ids)), dtype=torch.float32)
                target['sub_boxes']   = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes']   = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target['obj_labels']  = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['hoi_labels']  = torch.as_tensor(hoi_labels, dtype=torch.float32)
                target['sub_boxes']   = torch.stack(sub_boxes)
                target['obj_boxes']   = torch.stack(obj_boxes)

        else:
            target['filename'] = img_anno['file_name']
            target['boxes']    = boxes
            target['labels']   = classes
            target['id']       = idx

            if self._transforms is not None:
                img_0, _ = self._transforms[0](img, None)
                img, _   = self._transforms[1](img_0, None)

            clip_inputs = self.clip_preprocess(img_0)
            target['clip_inputs'] = clip_inputs

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'],
                             self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

        return img, target

    def set_rare_hois(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        counts = defaultdict(lambda: 0)
        for img_anno in annotations:
            hois   = img_anno['hoi_annotation']
            bboxes = img_anno['annotations']
            for hoi in hois:
                triplet = (
                    self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                    self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                    self._valid_verb_ids.index(hoi['category_id'])
                )
                counts[triplet] += 1

        self.rare_triplets     = []
        self.non_rare_triplets = []
        for triplet, count in counts.items():
            if count < 10:
                self.rare_triplets.append(triplet)
            else:
                self.non_rare_triplets.append(triplet)
        print("rare:{}, non-rare:{}".format(len(self.rare_triplets), len(self.non_rare_triplets)))

    def load_correct_mat(self, path):
        path = str(path)
        if path.endswith('.npy'):
            self.correct_mat = np.load(path)
        else:
            self.correct_mat = np.array(json.load(open(path)))


def make_hico_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return [T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            )]),
            normalize
        ]

    if image_set == 'val':
        return [T.Compose([
            T.RandomResize([800], max_size=1333),
        ]),
            normalize
        ]

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'

    PATHS = {
        'train': (
            root / 'train',
            root / 'annotations' / 'trainval_hico.json',
            root / 'clip_feats_pool' / 'train'
        ),
        'val': (
            root / 'val',
            root / 'annotations' / 'test_hico.json',
            root / 'clip_feats_pool' / 'val'
        ),
    }

    # Hỗ trợ cả .npy và .json
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'
    if not CORRECT_MAT_PATH.exists():
        CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.json'

    img_folder, anno_file, clip_feats_folder = PATHS[image_set]
    dataset = HICODetection(
        image_set, img_folder, anno_file, clip_feats_folder,
        transforms=make_hico_transforms(image_set),
        num_queries=args.num_queries, args=args
    )

    if image_set == 'val':
        dataset.set_rare_hois(PATHS['train'][1])
        dataset.load_correct_mat(CORRECT_MAT_PATH)

    return dataset