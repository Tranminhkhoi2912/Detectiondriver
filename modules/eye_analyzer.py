"""
eye_analyzer.py
---------------
Tính EAR (Eye Aspect Ratio) - chỉ số đánh giá mức độ mở/nhắm của mắt.

Công thức (Soukupová & Čech, 2016):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

Trong đó p1..p6 là 6 điểm landmark quanh mắt theo thứ tự:
    p1: khóe mắt ngoài/trong (một bên)
    p2, p3: điểm mí trên
    p4: khóe mắt còn lại
    p5, p6: điểm mí dưới

Khi mắt mở, khoảng cách dọc (tử số) lớn -> EAR lớn (~0.25-0.35)
Khi mắt nhắm, khoảng cách dọc gần 0 -> EAR nhỏ (~0.05-0.15)

Lưu ý: giá trị EAR tính ở đây chính là 1 trong 3 ĐẶC TRƯNG ĐẦU VÀO cho model
Machine Learning tự train (xem modules/ml_classifier.py) - không chỉ dùng cho
so sánh ngưỡng cố định.
"""

from scipy.spatial import distance as dist
import numpy as np


def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    eye_points: mảng 6 điểm (x, y) theo đúng thứ tự landmark đã định nghĩa
                trong face_detector.RIGHT_EYE_IDX / LEFT_EYE_IDX
    """
    # Khoảng cách dọc
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    # Khoảng cách ngang
    C = dist.euclidean(eye_points[0], eye_points[3])

    if C == 0:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return ear


def average_ear(left_eye_points: np.ndarray, right_eye_points: np.ndarray) -> float:
    """Lấy trung bình EAR 2 mắt để ổn định hơn (tránh nháy 1 mắt gây nhiễu)."""
    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)
    return (left_ear + right_ear) / 2.0
