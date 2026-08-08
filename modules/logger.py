"""
logger.py
---------
Ghi log các sự kiện buồn ngủ/ngủ gật ra file CSV để phân tích sau, và lưu
ảnh chụp màn hình tại thời điểm cảnh báo làm bằng chứng / dữ liệu đánh giá.

So với bản gốc, thêm 2 cột "source" (ML hoặc RULE) và "ml_confidence" để có
thể phân tích sau này model ML đang hoạt động tốt tới đâu so với rule-based
trên dữ liệu thực tế khi vận hành.
"""

import os
import csv
import time
from datetime import datetime

import cv2

from config import LOG_DIR, SNAPSHOT_DIR, EVENT_LOG_FILE, SAVE_SNAPSHOT_ON_ALERT


class EventLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        self._file_exists = os.path.exists(EVENT_LOG_FILE)
        self._last_logged_status = None
        self._last_log_time = 0.0
        self._min_log_interval_sec = 1.0  # tránh ghi lặp lại quá dày mỗi frame

        if not self._file_exists:
            with open(EVENT_LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "status", "ear", "mar", "source", "ml_confidence", "snapshot_path"])

    def log_event(self, status: str, ear: float, mar: float, frame=None, source: str = "RULE", ml_confidence: float = 0.0):
        """Ghi 1 dòng sự kiện. Chỉ ghi khi trạng thái thay đổi hoặc đã qua
        khoảng thời gian tối thiểu, để tránh file log phình to vô ích."""
        now = time.time()
        if status == self._last_logged_status and (now - self._last_log_time) < self._min_log_interval_sec:
            return

        self._last_logged_status = status
        self._last_log_time = now

        snapshot_path = ""
        if SAVE_SNAPSHOT_ON_ALERT and frame is not None and status in (
            "DROWSY_ALERT", "HEAD_DROP", "FATIGUE_WARNING"
        ):
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snapshot_path = os.path.join(SNAPSHOT_DIR, f"{status}_{timestamp_str}.jpg")
            cv2.imwrite(snapshot_path, frame)

        with open(EVENT_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                status,
                round(ear, 4),
                round(mar, 4),
                source,
                round(ml_confidence, 4),
                snapshot_path,
            ])
