# Ôn tập Kiểm tra thường xuyên 2 - Python

Bộ mã gồm 10 chương trình tương ứng 10 bài trong PDF. Thuật toán chính được tự cài đặt; scikit-learn chỉ dùng để đối chiếu hoặc hỗ trợ chia dữ liệu/đánh giá theo yêu cầu.

## 1. Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 2. Chuẩn bị dữ liệu

Tải các file từ UCI Machine Learning Repository và đặt đúng tên trong thư mục `data/`:

| Bài | File cần đặt trong `data/` | Dataset UCI |
|---|---|---|
| 1 | `zoo.data`, `auto-mpg.data` | Zoo, Auto MPG |
| 2 | `data_banknote_authentication.txt` | Banknote Authentication |
| 3 | `auto-mpg.data` | Auto MPG |
| 4, 7 | `abalone.data` | Abalone |
| 5 | `wdbc.data` | Breast Cancer Wisconsin (Diagnostic) |
| 8 | `agaricus-lepiota.data`, `house-votes-84.data` | Mushroom, Congressional Voting Records |
| 9 | `car.data` | Car Evaluation |
| 10 | `seeds_dataset.txt` | Seeds |

Trang dữ liệu: https://archive.ics.uci.edu/

Lưu ý:
- Auto MPG: file `auto-mpg.data`, dữ liệu phân tách bằng khoảng trắng, tên xe có dấu ngoặc kép.
- Banknote: `data_banknote_authentication.txt`, CSV không có header.
- Abalone: `abalone.data`, CSV không có header.
- WDBC: `wdbc.data`, CSV không có header; cột 2 là M/B.
- Mushroom: `agaricus-lepiota.data`, CSV không có header; cột đầu là nhãn.
- Voting: `house-votes-84.data`, CSV không có header; cột đầu là nhãn.
- Car Evaluation: `car.data`, CSV không có header.
- Seeds: `seeds_dataset.txt`, phân tách bằng khoảng trắng; 7 đặc trưng + nhãn.

## 3. Chạy từng bài

Chạy từ thư mục dự án:

```bash
python bai01_search.py
python bai02_perceptron.py
python bai03_linear_regression.py
python bai04_multiple_regression.py
python bai05_logistic_regression.py
python bai06_knn_classification.py
python bai07_knn_regression.py
python bai08_naive_bayes.py
python bai09_id3.py
python bai10_astar_greedy.py
```

## 4. Ghi chú học tập

- Bài 1: tìm kiếm nhị phân chỉ chạy trên dữ liệu đã sắp xếp theo đúng khóa tìm kiếm.
- Bài 2: nhãn được đổi `0 -> +1` (thật), `1 -> -1` (giả), đúng quy ước trong đề.
- Bài 3: công thức đóng và Gradient Descent được cài riêng; đánh giá trên test.
- Bài 4: dùng `np.linalg.solve`, không tính nghịch đảo ma trận trực tiếp.
- Bài 5: Min-Max chỉ fit trên train, sau đó áp dụng cho test; nhãn M=1, B=0.
- Bài 6: Iris dùng Stratified 5-fold; Wine so sánh chưa chuẩn hóa và z-score.
- Bài 7: Min-Max theo train; MAE được tính cho k=1..30.
- Bài 8: Naive Bayes bỏ qua `?` khi tính xác suất thủ công; bản sklearn coi `?` là một giá trị riêng.
- Bài 9: ID3 chọn thuộc tính có Information Gain cao nhất; giá trị chưa gặp trả về nhãn đa số ở nút.
- Bài 10: đồ thị k-NN được chuyển thành vô hướng; kiểm tra liên thông trước khi tìm đường.

Một số kết quả có thể khác nhẹ do cách xử lý hòa phiếu, thứ tự mẫu hoặc phiên bản thư viện. Hãy ghi lại kết quả thực tế khi chạy trên máy của bạn.
