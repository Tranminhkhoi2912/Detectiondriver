"""
head_pose.py
------------
Ước lượng góc nghiêng đầu (chủ yếu là pitch - gật/gục đầu) bằng cv2.solvePnP,
dựa trên một mô hình khuôn mặt 3D chuẩn (generic) và các điểm 2D tương ứng
lấy từ landmark thực tế trên frame.

Đây là module TÙY CHỌN (bật/tắt qua config.ENABLE_HEAD_POSE) - hữu ích để
phát hiện trường hợp tài xế gục đầu về phía trước mà mắt có thể vẫn hé mở,
điều mà chỉ riêng EAR không bắt được tốt.
"""

import cv2
import numpy as np

# Mô hình 3D khuôn mặt chuẩn (đơn vị tùy ý, chỉ cần đúng tỉ lệ tương đối)
# Thứ tự phải khớp với head_pose_points trả về từ face_detector
_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),        # nose_tip
    (0.0, -330.0, -65.0),   # chin
    (-225.0, 170.0, -135.0),  # left_eye_corner (thực chất là mắt trái người dùng)
    (225.0, 170.0, -135.0),   # right_eye_corner
    (-150.0, -150.0, -125.0),  # left_mouth_corner
    (150.0, -150.0, -125.0),   # right_mouth_corner
], dtype=np.float64)

_POINT_ORDER = ["nose_tip", "chin", "left_eye_corner", "right_eye_corner",
                "left_mouth_corner", "right_mouth_corner"]


def estimate_head_pitch(head_pose_points: dict, frame_shape) -> float:
    """
    head_pose_points: dict từ FaceDetector.get_head_pose_points
    frame_shape: (h, w, ...) của frame gốc

    Trả về góc pitch (độ). Giá trị dương/âm tùy quy ước solvePnP;
    quan trọng là ĐỘ LỆCH so với baseline lúc đầu thẳng, không phải giá trị tuyệt đối.
    """
    h, w = frame_shape[:2]

    image_points = np.array(
        [head_pose_points[name] for name in _POINT_ORDER],
        dtype=np.float64
    )

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # giả định không có méo ống kính

    success, rotation_vector, translation_vector = cv2.solvePnP(
        _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

    pitch = float(np.array(euler_angles).flatten()[0])
    return pitch
