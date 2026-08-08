"""
face_detector.py
-----------------
Phát hiện khuôn mặt và trích xuất landmark bằng MediaPipe Face Mesh.
MediaPipe được chọn thay cho dlib vì: nhẹ hơn, không cần tải file model .dat
riêng, chạy tốt trên CPU với tốc độ real-time.

Chỉ số landmark quan trọng (trong tổng số 468 điểm của Face Mesh):
- Mắt phải: 33, 160, 158, 133, 153, 144
- Mắt trái : 362, 385, 387, 263, 373, 380
- Miệng    : 61, 291 (khóe miệng), 13, 14 (môi trên/dưới giữa)
- Mũi/cằm  : dùng cho ước lượng head pose
"""

import cv2
import mediapipe as mp
import numpy as np


# Các bộ điểm landmark theo chuẩn 6-điểm (giống công thức EAR gốc của dlib 68 điểm)
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Điểm dùng để tính MAR (mouth aspect ratio)
MOUTH_IDX = {
    "left_corner": 61,
    "right_corner": 291,
    "top_lip": 13,
    "bottom_lip": 14,
}

# Điểm dùng để ước lượng head pose (solvePnP)
HEAD_POSE_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 263,
    "right_eye_corner": 33,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}


class FaceDetector:
    def __init__(self, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        """
        Nhận frame BGR (từ OpenCV), trả về:
            - landmarks_px: dict {idx: (x, y)} tọa độ pixel cho các điểm cần dùng,
              hoặc None nếu không phát hiện khuôn mặt.
            - all_landmarks: đối tượng landmark gốc của mediapipe (để vẽ debug nếu cần)
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None, None

        face_landmarks = results.multi_face_landmarks[0]

        needed_idx = set(RIGHT_EYE_IDX + LEFT_EYE_IDX + list(MOUTH_IDX.values()) + list(HEAD_POSE_IDX.values()))
        landmarks_px = {}
        for idx in needed_idx:
            lm = face_landmarks.landmark[idx]
            landmarks_px[idx] = (int(lm.x * w), int(lm.y * h))

        return landmarks_px, face_landmarks

    def get_eye_points(self, landmarks_px):
        right_eye = np.array([landmarks_px[i] for i in RIGHT_EYE_IDX], dtype=np.float64)
        left_eye = np.array([landmarks_px[i] for i in LEFT_EYE_IDX], dtype=np.float64)
        return left_eye, right_eye

    def get_mouth_points(self, landmarks_px):
        return {
            "left_corner": np.array(landmarks_px[MOUTH_IDX["left_corner"]], dtype=np.float64),
            "right_corner": np.array(landmarks_px[MOUTH_IDX["right_corner"]], dtype=np.float64),
            "top_lip": np.array(landmarks_px[MOUTH_IDX["top_lip"]], dtype=np.float64),
            "bottom_lip": np.array(landmarks_px[MOUTH_IDX["bottom_lip"]], dtype=np.float64),
        }

    def get_head_pose_points(self, landmarks_px):
        return {name: landmarks_px[idx] for name, idx in HEAD_POSE_IDX.items()}

    def close(self):
        self.face_mesh.close()
