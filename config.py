"""
config.py
---------
File cấu hình trung tâm. Chỉnh ngưỡng ở đây thay vì rải rác trong code.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== CAMERA ====================
CAMERA_INDEX = 0          # 0 = webcam mặc định. Đổi nếu có nhiều camera.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# ==================== EAR (Eye Aspect Ratio) ====================
# EAR càng nhỏ nghĩa là mắt càng nhắm. Người bình thường mắt mở EAR ~0.25-0.35
# Các ngưỡng dưới đây được dùng làm FALLBACK (rule-based) khi model ML
# chưa sẵn sàng hoặc chưa đủ tự tin để dự đoán.
EAR_THRESHOLD = 0.21          # Dưới ngưỡng này coi như mắt đang nhắm
EAR_CONSEC_FRAMES = 20        # Số frame liên tiếp mắt nhắm mới tính là "buồn ngủ"
                               # (~0.66s ở 30fps) - tránh báo động giả khi chớp mắt

# Ngưỡng cảnh báo "ngủ gật thật sự" (mắt nhắm lâu hơn)
DROWSY_ALERT_FRAMES = 45      # ~1.5s liên tục -> cảnh báo mạnh (còi)

# ==================== MAR (Mouth Aspect Ratio) - phát hiện ngáp ====================
MAR_THRESHOLD = 0.6
YAWN_CONSEC_FRAMES = 15
YAWN_COUNT_WINDOW_SEC = 60     # Đếm số lần ngáp trong 60s gần nhất
YAWN_COUNT_ALERT = 3           # Ngáp >= 3 lần trong khoảng trên -> cảnh báo mệt mỏi

# ==================== HEAD POSE (gục đầu) ====================
ENABLE_HEAD_POSE = True
HEAD_PITCH_DROP_THRESHOLD = 15  # độ lệch (degree) so với baseline coi là gục đầu
HEAD_POSE_CONSEC_FRAMES = 20

# ==================== CALIBRATION ====================
ENABLE_CALIBRATION = True
CALIBRATION_SECONDS = 5        # Thời gian hiệu chỉnh baseline EAR lúc mắt mở bình thường
CALIBRATION_EAR_MULTIPLIER = 0.75  # threshold cá nhân hóa = baseline_EAR * hệ số này

# ==================== MACHINE LEARNING (PHẦN TỰ TRAIN) ====================
# Bật/tắt việc dùng model ML đã tự huấn luyện thay cho ngưỡng cố định.
# Nếu model chưa tồn tại (chưa chạy scripts/train_model.py), hệ thống sẽ TỰ
# ĐỘNG fallback về logic rule-based ở trên, chương trình vẫn chạy bình thường.
USE_ML_MODEL = True

# Đường dẫn dữ liệu huấn luyện (do scripts/collect_data.py tạo ra, nhãn do
# NGƯỜI DÙNG tự gán thủ công lúc thu thập - không dùng lại nhãn rule-based).
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")

# Đường dẫn model đã train (do scripts/train_model.py tạo ra).
ML_MODEL_PATH = os.path.join(BASE_DIR, "models", "drowsiness_classifier.pkl")

# Xác suất tối thiểu để tin vào dự đoán của model; dưới ngưỡng này hệ thống
# fallback về rule-based cho frame đó (an toàn hơn khi model không chắc chắn).
ML_CONFIDENCE_THRESHOLD = 0.6

# ==================== ALERT (cảnh báo) ====================
ALERT_SOUND_PATH = os.path.join(BASE_DIR, "assets", "alarm.wav")
ALERT_COOLDOWN_SEC = 3          # Thời gian tối thiểu giữa 2 lần phát còi liên tiếp
ALERT_ESCALATE_AFTER_SEC = 8    # Nếu vẫn ngủ gật sau thời gian này -> cảnh báo cấp cao hơn

# ==================== LOGGING ====================
LOG_DIR = os.path.join(BASE_DIR, "logs")
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
EVENT_LOG_FILE = os.path.join(LOG_DIR, "events.csv")
SAVE_SNAPSHOT_ON_ALERT = True

# ==================== DEBUG / DISPLAY ====================
SHOW_DEBUG_OVERLAY = True
SHOW_FPS = True
SHOW_ML_INFO = True   # hiển thị nhãn + độ tin cậy của model ML trên màn hình
