# Implementation-of-HOICLIP-for-HOI-prediction

Dự án này triển khai và đánh giá mô hình **HOICLIP** (Human-Object Interaction Detection) trên tập dữ liệu video an ninh **MEVA (Multimodal Event Detection)** để nhận diện và định vị các hành động tương tác giữa người và phương tiện/vật thể (ví dụ: *boarding* - lên xe, *alighting* - xuống xe, *opening trunk* - đóng/mở cốp xe).

---

## 📌 1. Thông tin cơ bản (Overview)
- **Bài toán:** Nhận diện tương tác Người - Vật thể (Human-Object Interaction - HOI Detection).
- **Mục tiêu:** Phát hiện đồng thời bounding box của Con người (Subject), Bounding box của Vật thể (Object) và nhãn hành động (Verb/Action) liên kết giữa hai đối tượng đó trong từng khung hình video.
- **Kiến trúc cốt lõi:** Dựa trên mạng Transformer (tương tự kiến trúc DETR) kết hợp sức mạnh tiền huấn luyện đa phương thức của **CLIP** nhằm tối ưu hóa việc căn chỉnh đặc trưng thị giác và ngôn ngữ của hành động.

---

## 📊 2. Tập dữ liệu (Datasets)
Dự án tập trung huấn luyện và đánh giá trên tập dữ liệu camera an ninh giám sát quy mô lớn:
- **Tập dữ liệu chính:** **MEVA (Multimodal Event Detection)**.
- **Đặc điểm:** Định dạng bối cảnh góc quay rộng, camera tĩnh trên cao, bao quát các hoạt động giao thông tại bãi đỗ xe, ngã tư, tòa nhà công cộng. Các hành động tương tác như leo lên/xuống xe thường có mật độ thưa thớt (long-tail distribution).
- **Định dạng đánh giá:** Dữ liệu sau khi trích xuất được chuẩn hóa theo cấu trúc của tập dữ liệu HOI chuẩn thế giới **HICO-DET** để chạy qua bộ chấm điểm mAP.

---

## 🤖 3. Mô hình (Model & Paper)
Mô hình sử dụng trong mã nguồn này là **HOICLIP**, một trong những cấu trúc State-of-the-Art tận dụng tri thức từ mô hình CLIP để giải quyết bài toán HOI trong điều kiện zero-shot hoặc dữ liệu ít nhãn.

- **Link Paper gốc:** [HOICLIP: Efficient Knowledge Transfer for Human-Object Interaction Detection with Class-Injected CLIP](https://arxiv.org/abs/2211.16147)
- **Cơ chế hoạt động:** Mô hình trích xuất Object Queries, đẩy qua các tầng decoder để dự đoán đồng thời vị trí các box. Sau đó liên kết các box thông qua một danh sách ánh xạ chỉ mục dạng `subject_id` và `object_id` nằm trong trường dữ liệu tương tác độc lập.

---

## 💻 4. Cách sử dụng (Usage)

## ⚙️ Cài đặt dự án (Installation)

Để tải mã nguồn dự án và cài đặt các thư viện môi trường cần thiết, hãy thực hiện lần lượt các bước sau trong Terminal:

```bash
# 1. Clone (tải) kho mã nguồn từ GitHub về máy cục bộ
git clone [https://github.com/Silverz27/Implementation-of-HOICLIP-for-HOI-prediction.git](https://github.com/Silverz27/Implementation-of-HOICLIP-for-HOI-prediction.git)

# 2. Di chuyển vào thư mục dự án vừa tải về
cd Implementation-of-HOICLIP-for-HOI-prediction

# 3. Tạo một môi trường ảo Conda mới (Khuyến khích để tránh xung đột thư viện)
conda create -n hoiclip python=3.9 -y
conda activate hoiclip

# 4. Cài đặt các thư viện bổ trợ bắt buộc
pip install -r requirements.txt

### Chạy chế độ Đánh giá (Evaluation)
Để tiến hành test kiểm thử mô hình và xuất file log kết quả tính mAP từ checkpoint tốt nhất (`checkpoint_best.pth`), sử dụng lệnh sau trong Terminal:

```bash
python main.py \
    --dataset_file hico \
    --resume checkpoint_best.pth \
    --eval \
    --use_nms_filter \
    --json_file C:/Users/Thang/Downloads/HOICLIP/results.json
