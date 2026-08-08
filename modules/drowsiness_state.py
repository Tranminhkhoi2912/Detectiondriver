"""
drowsiness_state.py
--------------------
Máy trạng thái trung tâm: nhận các chỉ số (EAR, MAR, head pitch) mỗi frame,
tích lũy bộ đếm, và quyết định trạng thái hiện tại của tài xế.

Trạng thái đầu ra:
    AWAKE           - tỉnh táo bình thường
    EYE_WARNING     - mắt bắt đầu nhắm lâu hơn bình thường (chưa đủ ngưỡng cảnh báo mạnh)
    DROWSY_ALERT    - mắt nhắm đủ lâu -> NGỦ GẬT, cần cảnh báo ngay
    YAWNING         - đang ngáp
    FATIGUE_WARNING - ngáp nhiều lần trong khoảng thời gian ngắn -> dấu hiệu mệt mỏi
    HEAD_DROP       - gục đầu (nếu bật head pose)
    NO_FACE         - không phát hiện được khuôn mặt (tài xế quay đi / che khuất / camera lỗi)

--------------------------------------------------------------------------
PHẦN NÂNG CẤP (Machine Learning tự train):
--------------------------------------------------------------------------
Ở bản gốc, việc xác định "mắt đang nhắm hay không / đang ngáp hay không / có
đang gục đầu hay không" mỗi frame chỉ dựa vào SO SÁNH NGƯỠNG CỐ ĐỊNH
(rule-based, ví dụ EAR < 0.21).

Ở bản này, quyết định đó được giao cho một MODEL MACHINE LEARNING ĐÃ TỰ
HUẤN LUYỆN (RandomForestClassifier, xem modules/ml_classifier.py +
scripts/train_model.py), nhận 3 đặc trưng (EAR, MAR, pitch_dev) và tự học
ra ranh giới quyết định từ dữ liệu có nhãn thay vì con người tự đặt ngưỡng.

Nếu model chưa được train hoặc không đủ tự tin cho 1 frame cụ thể, hệ thống
tự động fallback về logic ngưỡng cố định để đảm bảo luôn hoạt động được.

Lớp bộ đếm frame liên tiếp (debounce) vẫn được giữ nguyên phía trên tín hiệu
mắt/miệng/đầu mỗi frame - dù tín hiệu đó đến từ ML hay rule-based - để tránh
báo động giả do chớp mắt/nói chuyện bình thường/nhiễu tức thời.
"""

import time
from collections import deque

import config
from modules.ml_classifier import MLClassifier


class DrowsinessState:
    def __init__(self):
        self.eye_closed_counter = 0
        self.yawn_counter = 0
        self.head_drop_counter = 0

        self.yawn_timestamps = deque()  # để đếm số lần ngáp trong cửa sổ thời gian
        self._was_yawning = False

        self.baseline_pitch = None  # thiết lập sau bước calibration

        self.current_status = "AWAKE"
        self.current_ear = 0.0
        self.current_mar = 0.0
        self.current_pitch_dev = 0.0

        # ---------- Machine Learning (tự train) ----------
        self.ml_classifier = MLClassifier() if config.USE_ML_MODEL else None
        self.using_ml = bool(self.ml_classifier and self.ml_classifier.is_ready)
        self.last_ml_label = None
        self.last_ml_confidence = 0.0

    def set_baseline_pitch(self, pitch: float):
        self.baseline_pitch = pitch

    def reset_no_face(self):
        """Khi mất khuôn mặt, không reset toàn bộ counter EAR ngay (có thể do
        chớp mắt kèm rung camera 1-2 frame), nhưng đánh dấu trạng thái NO_FACE."""
        self.current_status = "NO_FACE"

    def _get_frame_signals(self, ear: float, mar: float, pitch_dev):
        """
        Quyết định 3 tín hiệu nhị phân của frame hiện tại:
            eye_closed_now, yawning_now, head_drop_now

        Ưu tiên dùng model ML tự train nếu đã sẵn sàng (self.using_ml=True)
        VÀ đủ tự tin (confidence >= config.ML_CONFIDENCE_THRESHOLD).
        Ngược lại, dùng logic ngưỡng cố định (rule-based) làm fallback.
        """
        # ----- Fallback rule-based (luôn tính trước, dùng khi cần) -----
        eye_closed_now = ear < config.EAR_THRESHOLD
        yawning_now = mar > config.MAR_THRESHOLD
        head_drop_now = (pitch_dev is not None) and (pitch_dev > config.HEAD_PITCH_DROP_THRESHOLD)

        if self.using_ml:
            label, confidence = self.ml_classifier.predict(ear, mar, pitch_dev or 0.0)
            self.last_ml_label = label
            self.last_ml_confidence = confidence

            if label is not None and confidence >= config.ML_CONFIDENCE_THRESHOLD:
                # Model đủ tự tin -> dùng dự đoán của model thay cho rule-based
                eye_closed_now = (label == "EYE_CLOSED")
                yawning_now = (label == "YAWNING")
                head_drop_now = (label == "HEAD_DROP")
            # Nếu model không đủ tự tin -> giữ nguyên giá trị rule-based đã tính ở trên

        return eye_closed_now, yawning_now, head_drop_now

    def update(self, ear: float, mar: float, pitch: float = None) -> dict:
        """
        Cập nhật 1 frame dữ liệu. Trả về dict mô tả trạng thái để main.py
        quyết định hiển thị/cảnh báo:
            {
                "status": str,
                "ear": float,
                "mar": float,
                "eye_closed_frames": int,
                "should_alert_eye": bool,
                "should_alert_fatigue": bool,
                "should_alert_head": bool,
                "ml_label": str hoặc None,       # nhãn model ML dự đoán (nếu có)
                "ml_confidence": float,          # độ tin cậy của model (0-1)
                "source": "ML" hoặc "RULE",      # nguồn quyết định đang dùng
            }
        """
        self.current_ear = ear
        self.current_mar = mar

        pitch_dev = None
        if config.ENABLE_HEAD_POSE and pitch is not None and self.baseline_pitch is not None:
            pitch_dev = abs(pitch - self.baseline_pitch)
            self.current_pitch_dev = pitch_dev

        eye_closed_now, yawning_now, head_drop_now = self._get_frame_signals(ear, mar, pitch_dev)

        result = {
            "status": "AWAKE",
            "ear": ear,
            "mar": mar,
            "eye_closed_frames": 0,
            "should_alert_eye": False,
            "should_alert_fatigue": False,
            "should_alert_head": False,
            "ml_label": self.last_ml_label,
            "ml_confidence": self.last_ml_confidence,
            "source": "ML" if self.using_ml else "RULE",
        }

        # ---------- Phân tích mắt ----------
        if eye_closed_now:
            self.eye_closed_counter += 1
        else:
            self.eye_closed_counter = 0

        result["eye_closed_frames"] = self.eye_closed_counter

        if self.eye_closed_counter >= config.DROWSY_ALERT_FRAMES:
            result["status"] = "DROWSY_ALERT"
            result["should_alert_eye"] = True
        elif self.eye_closed_counter >= config.EAR_CONSEC_FRAMES:
            result["status"] = "EYE_WARNING"

        # ---------- Phân tích ngáp ----------
        if yawning_now:
            self.yawn_counter += 1
        else:
            # Nếu vừa kết thúc 1 chuỗi ngáp đủ dài -> tính là 1 lần ngáp hợp lệ
            if self.yawn_counter >= config.YAWN_CONSEC_FRAMES and not self._was_yawning:
                self.yawn_timestamps.append(time.time())
            self.yawn_counter = 0

        self._was_yawning = self.yawn_counter >= config.YAWN_CONSEC_FRAMES

        if self.yawn_counter >= config.YAWN_CONSEC_FRAMES and result["status"] == "AWAKE":
            result["status"] = "YAWNING"

        # Dọn các mốc ngáp quá cũ ngoài cửa sổ thời gian
        now = time.time()
        while self.yawn_timestamps and now - self.yawn_timestamps[0] > config.YAWN_COUNT_WINDOW_SEC:
            self.yawn_timestamps.popleft()

        if len(self.yawn_timestamps) >= config.YAWN_COUNT_ALERT:
            result["should_alert_fatigue"] = True
            if result["status"] == "AWAKE":
                result["status"] = "FATIGUE_WARNING"

        # ---------- Phân tích head pose (tùy chọn) ----------
        if config.ENABLE_HEAD_POSE:
            if head_drop_now:
                self.head_drop_counter += 1
            else:
                self.head_drop_counter = 0

            if self.head_drop_counter >= config.HEAD_POSE_CONSEC_FRAMES:
                result["should_alert_head"] = True
                # Gục đầu được ưu tiên hiển thị cao nếu chưa có cảnh báo mắt
                if result["status"] not in ("DROWSY_ALERT",):
                    result["status"] = "HEAD_DROP"

        self.current_status = result["status"]
        return result

    @property
    def yawn_count_in_window(self) -> int:
        return len(self.yawn_timestamps)
