"""
ml_classifier.py
-----------------
Bọc (wrap) một model Machine Learning ĐÃ ĐƯỢC TỰ HUẤN LUYỆN
(xem scripts/collect_data.py + scripts/train_model.py) để phân loại trạng thái
của tài xế mỗi frame, dựa trên 3 đặc trưng hình học đã tính từ landmark:

    - EAR (Eye Aspect Ratio)         -> đặc trưng mắt
    - MAR (Mouth Aspect Ratio)       -> đặc trưng miệng (ngáp)
    - pitch_dev (độ lệch góc đầu)    -> đặc trưng tư thế đầu

Model được huấn luyện bằng scikit-learn (RandomForestClassifier) trên dữ liệu
THU THẬP THỦ CÔNG, người dùng tự bấm phím gán nhãn trong lúc quay - KHÔNG
dùng lại nhãn do logic rule-based (drowsiness_state.py bản cũ) sinh ra, để
tránh việc model chỉ "học lại" chính cái ngưỡng cố định cũ (circular training,
không có giá trị học thuật).

Nếu chưa train model (chưa chạy scripts/train_model.py, file .pkl chưa tồn
tại), hệ thống sẽ TỰ ĐỘNG fallback về logic ngưỡng cố định trong
drowsiness_state.py - chương trình luôn chạy được, không bắt buộc phải có
model ngay từ đầu.
"""

import os
import numpy as np

import config

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False


class MLClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names = ["ear", "mar", "pitch_dev"]
        self.is_ready = False

        if not _JOBLIB_AVAILABLE:
            print("[MLClassifier] Thư viện 'joblib' chưa được cài (xem requirements.txt).")
            print("[MLClassifier] -> Dùng tạm logic ngưỡng cố định (rule-based).")
            return

        if not os.path.exists(config.ML_MODEL_PATH):
            print(f"[MLClassifier] Chưa tìm thấy model đã train tại: {config.ML_MODEL_PATH}")
            print("[MLClassifier] Hướng dẫn tự train model:")
            print("[MLClassifier]   1) python scripts/collect_data.py   (thu thập dữ liệu có nhãn thủ công)")
            print("[MLClassifier]   2) python scripts/train_model.py    (huấn luyện + đánh giá model)")
            print("[MLClassifier] -> Dùng tạm logic ngưỡng cố định (rule-based) cho đến khi có model.")
            return

        try:
            bundle = joblib.load(config.ML_MODEL_PATH)
            self.model = bundle["model"]
            self.label_encoder = bundle["label_encoder"]
            self.feature_names = bundle.get("feature_names", self.feature_names)
            self.is_ready = True
            print(f"[MLClassifier] Đã tải model tự train: {config.ML_MODEL_PATH}")
            print(f"[MLClassifier] Các nhãn model có thể dự đoán: {list(self.label_encoder.classes_)}")
        except Exception as e:
            print(f"[MLClassifier] Lỗi khi tải model ({e}) -> dùng tạm rule-based.")

    def predict(self, ear: float, mar: float, pitch_dev: float):
        """
        Dự đoán nhãn trạng thái của 1 frame dựa trên (ear, mar, pitch_dev).

        Trả về:
            (label: str hoặc None, confidence: float)
            - label=None nếu model chưa sẵn sàng -> nơi gọi tự fallback rule-based.
            - confidence là xác suất lớp được chọn (0.0 - 1.0), dùng để quyết
              định có "đủ tin tưởng" model hay không (xem config.ML_CONFIDENCE_THRESHOLD).
        """
        if not self.is_ready:
            return None, 0.0

        features = np.array([[ear, mar, pitch_dev]])
        probs = self.model.predict_proba(features)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        label = self.label_encoder.inverse_transform([best_idx])[0]

        return label, confidence
