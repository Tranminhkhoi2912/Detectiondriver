"""
camera.py
---------
Quản lý việc đọc frame từ webcam HOẶC từ file video có sẵn.

- Nếu bạn truyền `video_path` -> mở file video (dùng để test hệ thống bằng
  video có sẵn thay vì phải ngồi trước webcam).
- Nếu không truyền -> mở webcam như cũ.
"""

import os
import cv2
import time
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS


class Camera:
    def __init__(self, index=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT,
                 video_path=None):
        self.video_path = video_path
        self.is_video = video_path is not None

        if self.is_video:
            if not os.path.exists(video_path):
                raise RuntimeError(f"Không tìm thấy file video: {video_path}")
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Không thể mở file video: {video_path}")
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps_source = self.cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
            print(f"[Camera] Mở file video: {video_path} "
                  f"({self.total_frames} frame, {self.fps_source:.1f} fps)")
        else:
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Không thể mở webcam tại index {index}. "
                    f"Kiểm tra lại camera hoặc thử đổi CAMERA_INDEX trong config.py"
                )
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        self._prev_time = time.time()
        self._fps = 0.0

    def read(self):
        """Đọc 1 frame. Trả về (success, frame). Với webcam frame được lật
        ngang (mirror) để trực quan giống soi gương; với file video KHÔNG lật
        để giữ đúng cảnh gốc của video."""
        success, frame = self.cap.read()
        if not success:
            return False, None
        if not self.is_video:
            frame = cv2.flip(frame, 1)

        # Tính FPS thực tế
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            # Làm mượt bằng trung bình trượt đơn giản
            self._fps = self._fps * 0.9 + instant_fps * 0.1

        return True, frame

    @property
    def fps(self):
        return self._fps

    def release(self):
        self.cap.release()
