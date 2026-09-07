

## Cài đặt
1. THIẾT LẬP MÔI TRƯỜNG ẢO 
macos:
python3.11 -m venv venv
source venv/bin/activate
window:
python -m venv venv
.\venv\Scripts\Activate

2. CÀI ĐẶT CÁC THƯ VIỆN 
pip install -r requirements.txt

3. CHẠY DỰ ÁN
python3 main.py

## Quy trình sử dụng (3 bước)

### Bước 1 — Thu thập dữ liệu huấn luyện (có nhãn thủ công)

```bash
python scripts/collect_data.py
```

Trong lúc chạy, **bạn tự đóng vai** các trạng thái khác nhau trước camera và
bấm phím tương ứng để gán nhãn (chương trình sẽ ghi dữ liệu liên tục khi có
nhãn đang được chọn):

| Phím | Nhãn | Hành động cần làm trước camera |
|---|---|---|
| `1` | `AWAKE` | Ngồi bình thường, mắt mở, nhìn thẳng |
| `2` | `EYE_CLOSED` | Chủ động nhắm mắt lâu (giả vờ buồn ngủ) |
| `3` | `YAWNING` | Chủ động ngáp to, kéo dài vài giây |
| `4` | `HEAD_DROP` | Chủ động gục đầu về phía trước/xuống |
| `0` | (tạm dừng) | Dùng khi đổi tư thế, không muốn ghi nhầm nhãn |
| `b` | — | Hiệu chỉnh baseline góc đầu (nhìn thẳng, giữ yên ~1s) — **nên bấm trước khi thu `HEAD_DROP`** |
| `q` | — | Lưu và thoát |

**Vì sao phải tự gán nhãn thủ công?** Nếu dùng lại nhãn do logic ngưỡng cố
định cũ sinh ra để train, model sẽ chỉ "học lại" đúng cái ngưỡng đó — không có
giá trị học thuật (circular training). Việc người dùng chủ động đóng vai và tự
gán nhãn đảm bảo dữ liệu huấn luyện độc lập với luật cũ.

**Khuyến nghị:** thu thập ít nhất 300-500 dòng cho mỗi nhãn (~10-15 giây liên
tục ở 30fps), lặp lại ở nhiều góc/khoảng cách/ánh sáng khác nhau. Có thể chạy
script này nhiều lần — dữ liệu mới sẽ được nối thêm vào `data/training_data.csv`
(không ghi đè).

### Bước 2 — Huấn luyện model

```bash
python scripts/train_model.py
```

Script sẽ:
1. Tách 80% dữ liệu để train, 20% để test (đánh giá khách quan).
2. Huấn luyện `RandomForestClassifier` (200 cây, `class_weight="balanced"`
   để không thiên vị lớp có nhiều mẫu hơn).
3. In ra **classification report** (precision/recall/f1-score từng nhãn) và
   **cross-validation accuracy** (đánh giá độ ổn định qua nhiều lần chia dữ liệu).
4. Lưu 2 biểu đồ: `models/confusion_matrix.png` và `models/feature_importance.png`.
5. Lưu model đã train vào `models/drowsiness_classifier.pkl`.

### Bước 3 — Chạy hệ thống với model đã train

```bash
python main.py
```

Chương trình tự động phát hiện file `models/drowsiness_classifier.pkl` và dùng
model ML để phân loại mỗi frame. Màn hình sẽ hiển thị dòng "Nguồn quyết định:
ML | Dự đoán: ... (xx%)" để bạn theo dõi. Nếu model không đủ tự tin cho một
frame cụ thể (`ML_CONFIDENCE_THRESHOLD` trong `config.py`), hệ thống tự động
fallback sang rule-based cho frame đó.

- Nhấn **`q`** để thoát.
- Nhấn **`c`** để hiệu chỉnh lại ngưỡng cá nhân hóa (rule-based fallback) bất cứ lúc nào.

**Nếu chưa chạy bước 1-2**, `main.py` vẫn chạy bình thường bằng logic
ngưỡng cố định (rule-based), không bắt buộc phải có model ngay từ đầu.

## Cấu trúc dự án

```
drowsiness_detector/
├── main.py                    # Vòng lặp chính, điều phối toàn bộ hệ thống
├── config.py                  # Tất cả tham số cấu hình (rule-based + ML)
├── requirements.txt
├── README.md
├── modules/
│   ├── camera.py               # Đọc webcam, tính FPS
│   ├── face_detector.py        # MediaPipe Face Mesh -> landmark mắt/miệng/đầu
│   ├── eye_analyzer.py         # Tính EAR
│   ├── mouth_analyzer.py       # Tính MAR (phát hiện ngáp)
│   ├── head_pose.py            # Ước lượng góc gục đầu (solvePnP)
│   ├── ml_classifier.py        # [MỚI] Nạp & dùng model ML đã tự train
│   ├── drowsiness_state.py     # Máy trạng thái: kết hợp ML + fallback rule-based
│   ├── alert_system.py         # Phát âm thanh cảnh báo (pygame)
│   └── logger.py               # Ghi log CSV + snapshot (có thêm cột nguồn ML/RULE)
├── scripts/
│   ├── collect_data.py         #  Thu thập dữ liệu huấn luyện có nhãn thủ công
│   └── train_model.py          #  Huấn luyện + đánh giá model ML
├── data/
│   └── training_data.csv        # [MỚI] Dữ liệu huấn luyện (sinh ra sau bước 1)
├── models/
│   ├── drowsiness_classifier.pkl  # [MỚI] Model đã train (sinh ra sau bước 2)
│   ├── confusion_matrix.png       # [MỚI] Biểu đồ đánh giá (sinh ra sau bước 2)
│   └── feature_importance.png     # [MỚI] Biểu đồ đánh giá (sinh ra sau bước 2)
├── assets/                     # (tùy chọn) đặt file alarm.wav của bạn ở đây
├── logs/
│   ├── events.csv               # Log toàn bộ sự kiện theo thời gian
│   └── snapshots/                # Ảnh chụp tại thời điểm cảnh báo
```

## Cách hoạt động chi tiết

1. **Phát hiện khuôn mặt & landmark**: MediaPipe Face Mesh (468 điểm), lấy
   các điểm quanh mắt, miệng, và điểm mốc ước lượng góc đầu.
2. **Trích xuất đặc trưng (feature engineering)**: từ landmark tính ra 3 con
   số — EAR (mắt), MAR (miệng), pitch_dev (độ lệch góc đầu so với baseline).
3. **Phân loại trạng thái frame — đây là phần được nâng cấp ML**:
   - Nếu có model đã train và đủ tự tin: `model.predict_proba([ear, mar, pitch_dev])`
     trả về nhãn (AWAKE/EYE_CLOSED/YAWNING/HEAD_DROP) + độ tin cậy.
   - Nếu chưa có model hoặc độ tin cậy dưới ngưỡng: fallback so sánh ngưỡng cố định.
4. **Máy trạng thái (`DrowsinessState`)**: dùng bộ đếm frame liên tiếp (debounce)
   trên tín hiệu ở bước 3 để quyết định trạng thái cuối: `AWAKE`, `EYE_WARNING`,
   `DROWSY_ALERT`, `YAWNING`, `FATIGUE_WARNING`, `HEAD_DROP`, `NO_FACE`.
5. **Cảnh báo & ghi log**: phát âm thanh (có cooldown + cơ chế leo thang), ghi
   log CSV (kèm nguồn quyết định ML/RULE) và lưu ảnh snapshot làm bằng chứng.

## Tùy chỉnh tham số ML

Trong `config.py`:

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `USE_ML_MODEL` | Bật/tắt dùng model ML (nếu tắt, luôn dùng rule-based) | `True` |
| `ML_CONFIDENCE_THRESHOLD` | Độ tin cậy tối thiểu để tin dự đoán của model | `0.6` |
| `ML_MODEL_PATH` | Đường dẫn file model đã train | `models/drowsiness_classifier.pkl` |
| `TRAINING_DATA_PATH` | Đường dẫn dữ liệu huấn luyện | `data/training_data.csv` |

Các tham số ngưỡng rule-based cũ (`EAR_THRESHOLD`, `MAR_THRESHOLD`,
`HEAD_PITCH_DROP_THRESHOLD`...) vẫn được giữ nguyên, dùng làm fallback.

## Gợi ý viết báo cáo đồ án

Phần Machine Learning trong dự án này phù hợp trình bày trong báo cáo với các mục:

1. **Bài toán**: phân loại đa lớp (multi-class classification) từ 3 đặc trưng số.
2. **Thu thập & gán nhãn dữ liệu**: mô tả quy trình thủ công ở `collect_data.py`,
   nêu rõ số mẫu mỗi lớp, cách đảm bảo dữ liệu đa dạng (nhiều người/góc/ánh sáng).
3. **Model & huấn luyện**: RandomForestClassifier, lý do lựa chọn, siêu tham số.
4. **Đánh giá**: chèn `confusion_matrix.png`, bảng classification report,
   điểm cross-validation, `feature_importance.png` kèm giải thích đặc trưng
   nào quan trọng nhất.
5. **So sánh với baseline rule-based**: có thể chạy cả 2 chế độ
   (`config.USE_ML_MODEL = True/False`) trên cùng video test và so sánh số
   lần cảnh báo đúng/sai (đặc biệt liên quan vấn đề `HEAD_DROP` hay báo động
   giả do nhiễu `solvePnP`, đã quan sát được từ log thực tế của bản rule-based).

## Giới hạn & lưu ý khi triển khai thực tế

- Webcam thường (không hồng ngoại) hoạt động kém trong điều kiện thiếu sáng
  ban đêm. Với xe thực tế nên cân nhắc dùng camera IR.
- Model ML hiện tại chỉ dùng 3 đặc trưng số (EAR, MAR, pitch_dev) — không dùng
  ảnh trực tiếp, nên độ chính xác phụ thuộc nhiều vào chất lượng landmark từ
  MediaPipe. Muốn chính xác hơn có thể nâng cấp lên CNN học trực tiếp từ ảnh
  crop vùng mắt/miệng (cần dataset lớn hơn, xem gợi ý trong phần trao đổi).
- Kính râm sẽ cản MediaPipe phát hiện landmark mắt — cả model ML lẫn rule-based
  đều sẽ mất tác dụng trong trường hợp này.
- Model được train trên dữ liệu do chính người dùng tự thu thập — nếu chỉ có
  1 người quay, model có thể "overfit" theo khuôn mặt người đó, kém tổng quát
  với người khác. Nên thu thập từ nhiều người nếu muốn model dùng chung.
- Đây là hệ thống hỗ trợ cảnh báo, **không thay thế** các hệ thống an toàn
  chuyên dụng đã được kiểm định (DMS thương mại có cảm biến IR, đạt chuẩn ISO).


lệnh xóa data train + model
rm -f data/training_data.csv models/drowsiness_classifier.pkl models/confusion_matrix.png models/feature_importance.png

lệnh xóa data train
rm -f data/training_data.csv

lệnh xóa hình ảnh snapshot
rm -f logs/snapshots/*.jpg
