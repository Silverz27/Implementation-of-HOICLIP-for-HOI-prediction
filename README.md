# Implementation-of-HOICLIP-for-HOI-prediction

Dự án này triển khai và đánh giá mô hình **HOICLIP** (Human-Object Interaction Detection) trên tập dữ liệu video an ninh **MEVA (Multimodal Event Detection)** để nhận diện và định vị các hành động tương tác giữa người và phương tiện/vật thể (ví dụ: *boarding* - lên xe, *alighting* - xuống xe, *opening trunk* - đóng/mở cốp xe).

---

## 1. Thông tin cơ bản (Overview)
- **Bài toán:** Nhận diện tương tác Người - Vật thể (Human-Object Interaction - HOI Detection).
- **Mục tiêu:** Phát hiện đồng thời bounding box của Con người (Subject), Bounding box của Vật thể (Object) và nhãn hành động (Verb/Action) liên kết giữa hai đối tượng đó trong từng khung hình video.
- **Kiến trúc cốt lõi:** Dựa trên mạng Transformer (tương tự kiến trúc DETR) kết hợp sức mạnh tiền huấn luyện đa phương thức của **CLIP** nhằm tối ưu hóa việc căn chỉnh đặc trưng thị giác và ngôn ngữ của hành động.

---

## 2. Tập dữ liệu (Datasets)
Dự án tập trung huấn luyện và đánh giá trên tập dữ liệu camera an ninh giám sát quy mô lớn:
- **Tập dữ liệu chính:** **MEVA (Multimodal Event Detection)**.
- **Đặc điểm:** Định dạng bối cảnh góc quay rộng, camera tĩnh trên cao, bao quát các hoạt động giao thông tại bãi đỗ xe, ngã tư, tòa nhà công cộng. Các hành động tương tác như leo lên/xuống xe thường có mật độ thưa thớt (long-tail distribution).
- **Định dạng đánh giá:** Dữ liệu sau khi trích xuất được chuẩn hóa theo cấu trúc của tập dữ liệu HOI chuẩn thế giới **HICO-DET** để chạy qua bộ chấm điểm mAP.

---

## 3. Mô hình (Model & Paper)
Mô hình sử dụng trong mã nguồn này là **HOICLIP**, một trong những cấu trúc State-of-the-Art tận dụng tri thức từ mô hình CLIP để giải quyết bài toán HOI trong điều kiện zero-shot hoặc dữ liệu ít nhãn.

- **Link Paper gốc:** [HOICLIP: Efficient Knowledge Transfer for Human-Object Interaction Detection with Class-Injected CLIP](https://arxiv.org/abs/2211.16147)
- **Cơ chế hoạt động:** Mô hình trích xuất Object Queries, đẩy qua các tầng decoder để dự đoán đồng thời vị trí các box. Sau đó liên kết các box thông qua một danh sách ánh xạ chỉ mục dạng `subject_id` và `object_id` nằm trong trường dữ liệu tương tác độc lập.

---

## 4. Cách sử dụng (Usage)

### a. Clone (tải) kho mã nguồn từ GitHub về máy cục bộ
```bash
git clone [https://github.com/Silverz27/Implementation-of-HOICLIP-for-HOI-prediction.git](https://github.com/Silverz27/Implementation-of-HOICLIP-for-HOI-prediction.git)
```

### b. Di chuyển vào thư mục dự án vừa tải về
```bash
cd Implementation-of-HOICLIP-for-HOI-prediction
```

### c. Tạo một môi trường ảo Conda mới (Khuyến khích để tránh xung đột thư viện)
```bash
conda create -n hoiclip python=3.9 -y
conda activate hoiclip
```

### d. Cài đặt các thư viện bổ trợ bắt buộc
```bash
pip install -r requirements.txt
```

### Chạy chế độ Đánh giá (Evaluation)
Để tiến hành test kiểm thử mô hình và xuất file log kết quả tính mAP từ checkpoint tốt nhất (`checkpoint_best.pth`), sử dụng lệnh sau trong Terminal:

```bash
python main.py \
    --dataset_file hico \
    --resume checkpoint_best.pth \
    --eval \
    --use_nms_filter \
    --json_file C:/Users/Thang/Downloads/HOICLIP/results.json
```

## 5. Kết quả kỳ vọng (Expected Results)
Sau khi quá trình Evaluation kết thúc, mô hình sẽ trả về các kết quả định dạng sau:

- Hiển thị trên Terminal: Điểm số mAP tính toán theo chuẩn VOC 11 điểm (bao gồm mAP full, mAP rare, và mAP non-rare).

- File kết quả đầu ra (results.json): Chứa cấu trúc lưu trữ chuẩn hóa phục vụ việc vẽ bounding box (Visualize):

    predictions: Danh sách chứa tọa độ bbox thô của tất cả các thực thể được phát hiện.

    hoi_prediction: Danh sách chứa các liên kết tương tác bao gồm: subject_id (vị trí index của người), object_id (vị trí index của vật thể), category_id (ID của hành động) và score (độ tự tin của tương tác).

## 6. Hạn chế (Limitations)
- Hiện tượng Overfitting bối cảnh tĩnh: Do camera an ninh có góc quay cố định, mô hình dễ bị học vẹt (overfit) vào bối cảnh nền của tập Train, dẫn đến Precision giảm khi kiểm thử trên các góc camera mới.

- Lỗi nhiễu hộp bọc sát rìa ảnh: Nhánh Regression đôi khi tính toán ra các tọa độ vượt biên (tọa độ âm) khi mục tiêu nằm ở rìa trên của khung hình (đã được khắc phục tạm thời bằng hàm clip_preds_boxes).

- Mật độ phân phối class lệch: Các tương tác hiếm (như đóng/mở cốp, xuống xe) có số lượng mẫu rất ít, khiến per-class mAP của các hành động này bị kéo thấp.

## 7. Hướng phát triển (Future Work)
- Tối ưu hóa siêu tham số lọc trùng (NMS): Thực hiện tinh chỉnh sâu các ngưỡng IoU (thres_nms) để loại bỏ hoàn toàn các box rác trùng lặp đè lên nhau, giúp thu gọn dung lượng file log JSON.

- Áp dụng Stratified Split dữ liệu: Thay đổi chiến thuật chia tập dữ liệu Train/Test theo Video-level hoặc Scene-level nhằm ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (Data Leakage) giữa các frame ảnh liền kề.

- Tăng cường dữ liệu thời gian (Temporal Features): Tích hợp thêm các module xử lý chuỗi thời gian (như GRU, LSTM hoặc Temporal Transformer) để mô hình tận dụng thông tin luồng chuyển động của các frame trước đó, thay vì chỉ phân tích ảnh tĩnh độc lập.
