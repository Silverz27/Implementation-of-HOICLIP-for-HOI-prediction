# Implementation of HOICLIP for HOI Prediction

> Triển khai và đánh giá mô hình **HOICLIP** (Human-Object Interaction Detection) trên tập dữ liệu video an ninh **MEVA** (Multiview Extended Video with Activities) để nhận diện và định vị các hành động tương tác giữa người và phương tiện (ví dụ: *lên xe*, *xuống xe*, *mở/đóng cốp xe*, *mở/đóng cửa xe*).

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Tập dữ liệu](#2-tập-dữ-liệu)
3. [Mô hình](#3-mô-hình)
4. [Cài đặt](#4-cài-đặt)
5. [Sử dụng](#5-sử-dụng)
6. [Kết quả kỳ vọng](#6-kết-quả-kỳ-vọng)
7. [Hạn chế](#7-hạn-chế)
8. [Hướng phát triển](#8-hướng-phát-triển)

---

## 1. Tổng quan

| | |
|---|---|
| **Bài toán** | Nhận diện tương tác Người – Vật thể (Human-Object Interaction Detection) |
| **Mục tiêu** | Phát hiện đồng thời bounding box của *Người* (subject), bounding box của *Vật thể* (object), và nhãn *hành động* (verb) liên kết giữa hai đối tượng trong từng khung hình video |
| **Kiến trúc cốt lõi** | Transformer dạng DETR kết hợp tri thức tiền huấn luyện đa phương thức từ **CLIP**, giúp căn chỉnh không gian đặc trưng thị giác và ngôn ngữ của hành động |

Đầu ra cuối cùng của mô hình là một tập hợp các bộ ba:

```
⟨ bbox_person, bbox_object, action_label, confidence_score ⟩
```

---

## 2. Tập dữ liệu

Dự án sử dụng tập dữ liệu video giám sát an ninh quy mô lớn **MEVA (Multiview Extended Video with Activities)**.

- **Trang chủ dataset:** [mevadata.org](https://mevadata.org)
- **Bài báo gốc:** Corona et al., *"MEVA: A Large-Scale Multiview, Multimodal Video Dataset for Activity Detection"*, WACV 2021. [[PDF]](https://openaccess.thecvf.com/content/WACV2021/papers/Corona_MEVA_A_Large-Scale_Multiview_Multimodal_Video_Dataset_for_Activity_Detection_WACV_2021_paper.pdf)
- **Đặc điểm:** Góc quay rộng, camera tĩnh trên cao, bao quát các hoạt động giao thông tại bãi đỗ xe, ngã tư, khu vực tòa nhà công cộng. Các hành động tương tác như lên/xuống xe có mật độ thưa (long-tail distribution).
- **Định dạng đánh giá:** Dữ liệu sau khi trích xuất được chuẩn hóa theo cấu trúc annotation của tập **HICO-DET** để có thể chạy trực tiếp qua bộ chấm điểm mAP chuẩn của cộng đồng HOI Detection.

### Các lớp hành động sử dụng

| ID | Tên hành động (MEVA) | Nhãn tiếng Việt |
|:--:|---|---|
| 1 | `person_enters_vehicle` | Lên xe |
| 2 | `person_exits_vehicle` | Xuống xe |
| 3 | `person_opens_vehicle_door` | Mở cửa xe |
| 4 | `person_closes_vehicle_door` | Đóng cửa xe |
| 5 | `person_opens_trunk` | Mở cốp xe |
| 6 | `person_closes_trunk` | Đóng cốp xe |
| 7 | `vehicle_stops` | Xe dừng |
| 8 | `vehicle_starts` | Xe di chuyển |

---

## 3. Mô hình

Mô hình sử dụng là **HOICLIP**, một trong những kiến trúc State-of-the-Art tận dụng tri thức từ CLIP để giải quyết bài toán HOI Detection trong điều kiện dữ liệu ít nhãn hoặc zero-shot.

- **Bài báo gốc:** Ning et al., *"HOICLIP: Efficient Knowledge Transfer for Human-Object Interaction Detection with Vision-Language Models"*, CVPR 2023. [[arXiv:2211.16147]](https://arxiv.org/abs/2211.16147)
- **Cơ chế hoạt động:** Mô hình trích xuất *Object Queries* / *Human Queries*, đưa qua các tầng decoder để dự đoán đồng thời vị trí các bounding box. Các box sau đó được liên kết với nhau thông qua một bảng ánh xạ chỉ mục dạng `subject_id` và `object_id`, được lưu trong trường dữ liệu tương tác (`hoi_annotation`) độc lập với trường bounding box.

---

## 4. Cài đặt

### 4.1. Clone kho mã nguồn

```bash
git clone https://github.com/Silverz27/Implementation-of-HOICLIP-for-HOI-prediction.git
cd Implementation-of-HOICLIP-for-HOI-prediction
```

### 4.2. Tạo môi trường ảo

Khuyến khích sử dụng Conda để tránh xung đột thư viện:

```bash
conda create -n hoiclip python=3.9 -y
conda activate hoiclip
```

### 4.3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4.4. Tải checkpoint và pretrained weights

Toàn bộ checkpoint được lưu tại Google Drive:
[**HOICLIP_Checkpoints**](https://drive.google.com/drive/folders/1uZ41TnfEr0GVoowvLv1eGiOQzp22qpKh?usp=drive_link)

Thư mục bao gồm:

| File | Mô tả |
|---|---|
| `detr_r50.pth` | Pretrained DETR backbone — dùng nếu muốn train mô hình từ đầu |
| `checkpoint_last.pth` | Checkpoint đã train qua 30 epochs — dùng để **resume** training |
| `checkpoint_best.pth` | Checkpoint có kết quả tốt nhất — dùng để **evaluate** |

Sau khi tải xong, di chuyển các file vào thư mục `train_results/`:

```bash
mkdir -p train_results
mv detr_r50.pth checkpoint_last.pth checkpoint_best.pth train_results/
```

---

## 5. Sử dụng

### 5.1. Huấn luyện (Train)

```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
  --dataset_file hico \
  --hoi_path /path/to/meva-datasets \
  --num_obj_classes 3 \
  --num_verb_classes 8 \
  --backbone resnet50 \
  --num_queries 64 \
  --dec_layers 3 \
  --epochs 30 \
  --lr_drop 20 \
  --batch_size 4 \
  --num_workers 4 \
  --pretrained train_results/detr_r50.pth \
  --with_clip_label \
  --with_obj_clip_label \
  --use_nms_filter \
  --output_dir train_results
```

> Để tiếp tục huấn luyện từ checkpoint có sẵn, thay `--pretrained` bằng `--resume train_results/checkpoint_last.pth`.

### 5.2. Đánh giá (Evaluation)

```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
  --resume train_results/checkpoint_best.pth \
  --dataset_file hico \
  --hoi_path /home/t4-vkist/.cache/kagglehub/datasets/sliverz/meva-datasets/versions/1 \
  --num_obj_classes 3 \
  --num_verb_classes 8 \
  --backbone resnet50 \
  --num_queries 64 \
  --dec_layers 3 \
  --eval \
  --num_workers 2 \
  --with_obj_clip_label \
  --use_nms_filter \
  --output_dir eval_results
```

---

## 6. Kết quả kỳ vọng

Sau khi quá trình evaluation kết thúc:

- **Terminal:** hiển thị điểm số mAP tính theo chuẩn VOC 11-point, bao gồm `mAP full`, `mAP rare`, và `mAP non-rare`.
- **File `results.json`:** chứa cấu trúc kết quả chuẩn hóa, phục vụ trực quan hóa (visualize):

  | Trường | Mô tả |
  |---|---|
  | `predictions` | Danh sách tọa độ bbox thô của tất cả thực thể được phát hiện |
  | `hoi_prediction` | Danh sách liên kết tương tác, gồm `subject_id` (index người), `object_id` (index vật thể), `category_id` (ID hành động) và `score` (độ tin cậy) |

---

## 7. Hạn chế

- **Overfitting bối cảnh tĩnh:** camera an ninh có góc quay cố định, mô hình dễ học vẹt bối cảnh nền của tập train, khiến precision giảm khi kiểm thử trên góc camera mới.
- **Lỗi bounding box sát rìa ảnh:** nhánh regression đôi khi cho ra tọa độ âm khi mục tiêu nằm sát rìa trên khung hình (đã khắc phục tạm thời bằng hàm `clip_preds_boxes`).
- **Phân phối lớp lệch (long-tail):** các tương tác hiếm (đóng/mở cốp, xuống xe) có số mẫu rất ít, khiến per-class mAP của các hành động này bị kéo thấp.

---

## 8. Hướng phát triển

- **Tối ưu siêu tham số NMS:** tinh chỉnh sâu ngưỡng IoU (`thres_nms`) để loại bỏ box trùng lặp, giảm dung lượng file log JSON.
- **Stratified Split theo Video/Scene:** thay đổi chiến lược chia tập train/test theo cấp video hoặc scene để ngăn rò rỉ dữ liệu (data leakage) giữa các frame liền kề.
- **Tích hợp đặc trưng thời gian:** bổ sung module xử lý chuỗi (GRU, LSTM, hoặc Temporal Transformer) để mô hình tận dụng thông tin chuyển động từ các frame trước, thay vì chỉ phân tích ảnh tĩnh độc lập.

---

## Tham khảo

1. S. Ning, L. Qiu, Y. Liu, X. He. *"HOICLIP: Efficient Knowledge Transfer for Human-Object Interaction Detection with Vision-Language Models"*. CVPR 2023.
2. K. Corona, K. Osterdahl, R. Collins, A. Hoogs. *"MEVA: A Large-Scale Multiview, Multimodal Video Dataset for Activity Detection"*. WACV 2021.
