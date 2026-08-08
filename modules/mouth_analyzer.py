"""
mouth_analyzer.py
------------------
Tính MAR (Mouth Aspect Ratio) để phát hiện ngáp - một dấu hiệu sớm
của sự mệt mỏi/buồn ngủ, thường xuất hiện trước khi mắt bắt đầu nhắm lâu.

MAR = khoảng_cách_dọc_môi / khoảng_cách_ngang_miệng

Miệng ngáp: khoảng cách dọc lớn -> MAR lớn (thường > 0.6)
Miệng bình thường/nói chuyện: MAR dao động thấp hơn và không kéo dài.
"""

from scipy.spatial import distance as dist


def mouth_aspect_ratio(mouth_points: dict) -> float:
    """
    mouth_points: dict có 4 khóa 'left_corner', 'right_corner', 'top_lip', 'bottom_lip'
                  (đầu ra của FaceDetector.get_mouth_points)
    """
    vertical = dist.euclidean(mouth_points["top_lip"], mouth_points["bottom_lip"])
    horizontal = dist.euclidean(mouth_points["left_corner"], mouth_points["right_corner"])

    if horizontal == 0:
        return 0.0

    mar = vertical / horizontal
    return mar
