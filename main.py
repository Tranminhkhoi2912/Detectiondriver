"""
main.py
-------
Điểm khởi chạy hệ thống nhận diện tài xế ngủ gật qua webcam.

Luồng xử lý mỗi frame:
    1. Đọc frame từ webcam
    2. Phát hiện khuôn mặt + landmark (MediaPipe Face Mesh)
    3. Tính EAR (mắt), MAR (miệng), pitch đầu (tùy chọn)
    4. Đưa vào máy trạng thái DrowsinessState -> nhận trạng thái hiện tại
       (bên trong sẽ tự dùng model ML đã tự train nếu có, hoặc rule-based
       nếu chưa train model - xem modules/drowsiness_state.py)
    5. Nếu cần cảnh báo -> phát âm thanh + ghi log + lưu ảnh
    6. Vẽ overlay debug (bao gồm cả thông tin model ML đang dùng) và hiển thị

Nhấn 'q' để thoát, 'c' để hiệu chỉnh lại (calibrate) EAR bất kỳ lúc nào.

Ghi chú: để dùng model ML tự train, chạy trước:
    python scripts/collect_data.py   (thu thập dữ liệu có nhãn thủ công)
    python scripts/train_model.py    (huấn luyện + đánh giá model)
Nếu chưa làm bước trên, chương trình vẫn chạy bình thường bằng rule-based.
"""

import sys
import time
import cv2
import numpy as np

import config
from modules.camera import Camera
from modules.face_detector import FaceDetector
from modules.eye_analyzer import average_ear
from modules.mouth_analyzer import mouth_aspect_ratio
from modules.head_pose import estimate_head_pitch
from modules.drowsiness_state import DrowsinessState
from modules.alert_system import AlertSystem
from modules.logger import EventLogger


STATUS_COLORS = {
    "AWAKE": (0, 200, 0),
    "EYE_WARNING": (0, 165, 255),
    "DROWSY_ALERT": (0, 0, 255),
    "YAWNING": (0, 200, 255),
    "FATIGUE_WARNING": (0, 140, 255),
    "HEAD_DROP": (0, 0, 255),
    "NO_FACE": (128, 128, 128),
}

STATUS_LABELS_VI = {
    "AWAKE": "TINH TAO",
    "EYE_WARNING": "MAT NHAM - CANH BAO",
    "DROWSY_ALERT": "NGU GAT !!!",
    "YAWNING": "DANG NGAP",
    "FATIGUE_WARNING": "MET MOI (NGAP NHIEU LAN)",
    "HEAD_DROP": "GUC DAU !!!",
    "NO_FACE": "KHONG THAY KHUON MAT",
}


def run_calibration(camera: Camera, detector: FaceDetector, state: DrowsinessState):
    """Thu thập EAR/pitch baseline trong vài giây đầu khi mắt mở bình thường,
    để cá nhân hóa ngưỡng thay vì dùng ngưỡng cố định cho mọi người.
    (Áp dụng cho cả trường hợp fallback rule-based khi model ML không đủ tin cậy.)"""
    print(f"[Calibration] Vui lòng nhìn thẳng vào camera, giữ mắt mở bình thường trong {config.CALIBRATION_SECONDS}s...")

    ear_samples = []
    pitch_samples = []
    start = time.time()

    while time.time() - start < config.CALIBRATION_SECONDS:
        success, frame = camera.read()
        if not success:
            continue

        landmarks_px, _ = detector.process(frame)
        remaining = config.CALIBRATION_SECONDS - (time.time() - start)

        if landmarks_px is not None:
            left_eye, right_eye = detector.get_eye_points(landmarks_px)
            ear = average_ear(left_eye, right_eye)
            ear_samples.append(ear)

            if config.ENABLE_HEAD_POSE:
                head_points = detector.get_head_pose_points(landmarks_px)
                pitch = estimate_head_pitch(head_points, frame.shape)
                pitch_samples.append(pitch)

        cv2.putText(frame, f"Dang hieu chinh... {remaining:.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if ear_samples:
        baseline_ear = float(np.median(ear_samples))
        personalized_threshold = baseline_ear * config.CALIBRATION_EAR_MULTIPLIER
        config.EAR_THRESHOLD = personalized_threshold
        print(f"[Calibration] Baseline EAR = {baseline_ear:.3f} -> Ngưỡng cảnh báo mới = {personalized_threshold:.3f}")
    else:
        print("[Calibration] Không phát hiện được khuôn mặt trong lúc hiệu chỉnh, dùng ngưỡng mặc định.")

    if pitch_samples:
        baseline_pitch = float(np.median(pitch_samples))
        state.set_baseline_pitch(baseline_pitch)
        print(f"[Calibration] Baseline head pitch = {baseline_pitch:.1f} độ")


def draw_overlay(frame, status: str, ear: float, mar: float, fps: float, yawn_count: int,
                  source: str = "RULE", ml_label=None, ml_confidence: float = 0.0):
    color = STATUS_COLORS.get(status, (255, 255, 255))
    label = STATUS_LABELS_VI.get(status, status)

    h, w = frame.shape[:2]

    # Khung viền màu theo trạng thái (cảnh báo trực quan toàn màn hình)
    if status in ("DROWSY_ALERT", "HEAD_DROP"):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)

    cv2.putText(frame, label, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if config.SHOW_DEBUG_OVERLAY:
        cv2.putText(frame, f"EAR: {ear:.3f} (nguong: {config.EAR_THRESHOLD:.3f})", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.3f}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"So lan ngap ({config.YAWN_COUNT_WINDOW_SEC}s): {yawn_count}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if config.SHOW_ML_INFO:
        ml_text = f"Nguon quyet dinh: {source}"
        if source == "ML" and ml_label is not None:
            ml_text += f" | Du doan: {ml_label} ({ml_confidence*100:.0f}%)"
        ml_color = (0, 255, 255) if source == "ML" else (180, 180, 180)
        cv2.putText(frame, ml_text, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ml_color, 1)

    if config.SHOW_FPS:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    return frame


def main():
    camera = Camera()
    detector = FaceDetector()
    state = DrowsinessState()
    alert_system = AlertSystem()
    event_logger = EventLogger()

    if state.using_ml:
        print("[Main] Đang dùng MODEL MACHINE LEARNING TỰ TRAIN để phân loại trạng thái.")
    else:
        print("[Main] Đang dùng logic NGƯỠNG CỐ ĐỊNH (rule-based). "
              "Chạy scripts/collect_data.py + scripts/train_model.py để bật model ML.")

    try:
        if config.ENABLE_CALIBRATION:
            run_calibration(camera, detector, state)

        print("[Main] Bắt đầu giám sát. Nhấn 'q' để thoát, 'c' để hiệu chỉnh lại.")

        while True:
            success, frame = camera.read()
            if not success:
                print("[Main] Không đọc được frame từ webcam, dừng chương trình.")
                break

            landmarks_px, _ = detector.process(frame)

            if landmarks_px is None:
                state.reset_no_face()
                frame = draw_overlay(frame, "NO_FACE", 0.0, 0.0, camera.fps, state.yawn_count_in_window,
                                      source="ML" if state.using_ml else "RULE")
                event_logger.log_event("NO_FACE", 0.0, 0.0, frame)
            else:
                left_eye, right_eye = detector.get_eye_points(landmarks_px)
                ear = average_ear(left_eye, right_eye)

                mouth_points = detector.get_mouth_points(landmarks_px)
                mar = mouth_aspect_ratio(mouth_points)

                pitch = None
                if config.ENABLE_HEAD_POSE:
                    head_points = detector.get_head_pose_points(landmarks_px)
                    pitch = estimate_head_pitch(head_points, frame.shape)

                result = state.update(ear, mar, pitch)

                if result["should_alert_eye"] or result["should_alert_head"]:
                    severity = "escalated" if state.eye_closed_counter > config.DROWSY_ALERT_FRAMES * 2 else "normal"
                    alert_system.trigger(severity)
                elif result["should_alert_fatigue"]:
                    alert_system.trigger("normal")
                else:
                    if result["status"] == "AWAKE":
                        alert_system.reset()

                event_logger.log_event(
                    result["status"], ear, mar, frame,
                    source=result["source"], ml_confidence=result["ml_confidence"],
                )

                frame = draw_overlay(
                    frame, result["status"], ear, mar, camera.fps, state.yawn_count_in_window,
                    source=result["source"], ml_label=result["ml_label"], ml_confidence=result["ml_confidence"],
                )

            cv2.imshow("Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                run_calibration(camera, detector, state)

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        print("[Main] Đã dừng hệ thống.")


if __name__ == "__main__":
    main()
