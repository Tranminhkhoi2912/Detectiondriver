"""
scripts/collect_data.py
------------------------
CÔNG CỤ THU THẬP DỮ LIỆU HUẤN LUYỆN (bước 1 của "tự train model").

QUAN TRỌNG: script này KHÔNG dùng lại nhãn do logic rule-based (ngưỡng cố
định) sinh ra, vì như vậy model sẽ chỉ "học lại" chính cái ngưỡng cũ - không
có giá trị học thuật (circular training). Thay vào đó, CHÍNH BẠN sẽ tự gán
nhãn bằng phím bấm trong lúc quay video thật trước webcam:

    Phím 1 -> đang gán nhãn AWAKE       (tỉnh táo, mắt mở bình thường, nhìn thẳng)
    Phím 2 -> đang gán nhãn EYE_CLOSED  (chủ động nhắm mắt lâu, giả buồn ngủ)
    Phím 3 -> đang gán nhãn YAWNING     (chủ động ngáp to, kéo dài)
    Phím 4 -> đang gán nhãn HEAD_DROP   (chủ động gục đầu về phía trước/xuống)
    Phím b -> hiệu chỉnh baseline góc đầu (nhìn thẳng vào camera, giữ yên ~1s)
    Phím 0 -> tạm dừng ghi (không gán nhãn nào) - dùng lúc đổi tư thế
    Phím q -> dừng và lưu file

Mỗi frame trong lúc ĐANG chọn 1 nhãn (khác None) sẽ được ghi 1 dòng vào CSV
gồm 3 đặc trưng (EAR, MAR, pitch_dev) + nhãn hiện tại.

Khuyến nghị thu thập để model học tốt:
    - Mỗi nhãn nên có ít nhất 300-500 dòng (khoảng 10-15 giây liên tục ở ~30fps)
    - Lặp lại nhiều lần, đổi khoảng cách/góc nhìn/ánh sáng khác nhau
    - Nếu có thể, nhờ thêm vài người khác nhau quay giúp để model tổng quát hơn,
      tránh chỉ học "khuôn mặt của một người"
    - Nhớ bấm phím 'b' để hiệu chỉnh baseline góc đầu TRƯỚC khi thu thập
      nhãn HEAD_DROP, nếu không pitch_dev sẽ luôn tính từ baseline mặc định = 0
"""

import os
import sys
import csv

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.camera import Camera
from modules.face_detector import FaceDetector
from modules.eye_analyzer import average_ear
from modules.mouth_analyzer import mouth_aspect_ratio
from modules.head_pose import estimate_head_pitch


LABELS = {
    ord('1'): "AWAKE",
    ord('2'): "EYE_CLOSED",
    ord('3'): "YAWNING",
    ord('4'): "HEAD_DROP",
}

LABEL_COLORS = {
    "AWAKE": (0, 200, 0),
    "EYE_CLOSED": (0, 0, 255),
    "YAWNING": (0, 200, 255),
    "HEAD_DROP": (0, 140, 255),
    None: (150, 150, 150),
}


def main():
    os.makedirs(os.path.dirname(config.TRAINING_DATA_PATH), exist_ok=True)
    file_exists = os.path.exists(config.TRAINING_DATA_PATH)

    csv_file = open(config.TRAINING_DATA_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["ear", "mar", "pitch_dev", "label"])

    camera = Camera()
    detector = FaceDetector()

    current_label = None
    baseline_pitch = None
    row_counts = {"AWAKE": 0, "EYE_CLOSED": 0, "YAWNING": 0, "HEAD_DROP": 0}

    print("=" * 70)
    print(" THU THẬP DỮ LIỆU HUẤN LUYỆN CHO MODEL MACHINE LEARNING")
    print("=" * 70)
    print(" 1 = AWAKE       2 = EYE_CLOSED     3 = YAWNING     4 = HEAD_DROP")
    print(" 0 = Tạm dừng ghi (bỏ chọn nhãn)     b = Hiệu chỉnh baseline góc đầu")
    print(" q = Lưu và thoát")
    print("=" * 70)
    print(" Hãy bấm 'b' để hiệu chỉnh baseline đầu trước khi thu thập HEAD_DROP!")
    print("=" * 70)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                continue

            landmarks_px, _ = detector.process(frame)

            if landmarks_px is not None:
                left_eye, right_eye = detector.get_eye_points(landmarks_px)
                ear = average_ear(left_eye, right_eye)

                mouth_points = detector.get_mouth_points(landmarks_px)
                mar = mouth_aspect_ratio(mouth_points)

                pitch_dev = 0.0
                if config.ENABLE_HEAD_POSE:
                    head_points = detector.get_head_pose_points(landmarks_px)
                    pitch = estimate_head_pitch(head_points, frame.shape)
                    if baseline_pitch is not None:
                        pitch_dev = abs(pitch - baseline_pitch)

                if current_label is not None:
                    writer.writerow([round(ear, 4), round(mar, 4), round(pitch_dev, 4), current_label])
                    row_counts[current_label] += 1

                color = LABEL_COLORS.get(current_label, (255, 255, 255))
                cv2.putText(frame, f"EAR:{ear:.3f}  MAR:{mar:.3f}  PitchDev:{pitch_dev:.1f}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"NHAN HIEN TAI: {current_label or '(tam dung)'}",
                            (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if baseline_pitch is None:
                    cv2.putText(frame, "CHUA HIEU CHINH BASELINE DAU - bam 'b'",
                                (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

                y = 115
                for k, v in row_counts.items():
                    cv2.putText(frame, f"{k}: {v} mau", (20, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    y += 22
            else:
                cv2.putText(frame, "KHONG THAY KHUON MAT", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Thu thap du lieu huan luyen", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('0'):
                current_label = None
                print("[Nhãn] Tạm dừng ghi.")
            elif key in LABELS:
                current_label = LABELS[key]
                print(f"[Nhãn] Chuyển sang: {current_label}")
            elif key == ord('b') and landmarks_px is not None and config.ENABLE_HEAD_POSE:
                head_points = detector.get_head_pose_points(landmarks_px)
                baseline_pitch = estimate_head_pitch(head_points, frame.shape)
                print(f"[Baseline] Đã hiệu chỉnh pitch gốc = {baseline_pitch:.1f} độ")

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        csv_file.close()

        total = sum(row_counts.values())
        print("\n" + "=" * 70)
        print(f"Đã lưu {total} dòng dữ liệu mới vào: {config.TRAINING_DATA_PATH}")
        for k, v in row_counts.items():
            flag = "  (CÒN ÍT, nên thu thập thêm)" if v < 200 else ""
            print(f"  - {k}: {v} dòng{flag}")
        print("Tiếp theo, chạy: python scripts/train_model.py")
        print("=" * 70)


if __name__ == "__main__":
    main()
