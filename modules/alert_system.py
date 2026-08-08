"""
alert_system.py
----------------
Quản lý phát âm thanh cảnh báo. Dùng pygame.mixer vì nhẹ, không block luồng
chính lâu, và cho phép kiểm tra "đang phát hay không" để tránh chồng âm thanh.

Nếu không tìm thấy file âm thanh (assets/alarm.wav), hệ thống sẽ tự tạo
một tiếng beep đơn giản bằng cách sinh sóng sine (không cần file ngoài),
đảm bảo demo chạy được ngay cả khi chưa có file âm thanh riêng.
"""

import os
import time
import numpy as np

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

from config import ALERT_SOUND_PATH, ALERT_COOLDOWN_SEC, ALERT_ESCALATE_AFTER_SEC


def _generate_beep_array(freq=1000, duration_sec=0.5, sample_rate=44100, volume=0.5):
    """Sinh mảng sóng sine làm âm beep dự phòng khi không có file .wav."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    tone = np.sin(freq * t * 2 * np.pi)
    audio = (tone * volume * 32767).astype(np.int16)
    stereo = np.repeat(audio.reshape(-1, 1), 2, axis=1)
    return stereo


class AlertSystem:
    def __init__(self):
        self.enabled = _PYGAME_AVAILABLE
        self._sound = None
        self._escalated_sound = None
        self._last_alert_time = 0.0
        self._alert_start_time = None  # thời điểm bắt đầu chuỗi cảnh báo hiện tại

        if not self.enabled:
            print("[AlertSystem] Cảnh báo: pygame chưa được cài, sẽ chỉ cảnh báo bằng hiển thị màn hình.")
            return

        pygame.mixer.init(frequency=44100, size=-16, channels=2)

        if os.path.exists(ALERT_SOUND_PATH):
            self._sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
        else:
            # Fallback: tự sinh âm beep, không cần file ngoài
            self._sound = pygame.sndarray.make_sound(_generate_beep_array(freq=900, duration_sec=0.4))
            self._escalated_sound = pygame.sndarray.make_sound(_generate_beep_array(freq=1400, duration_sec=0.6))

    def trigger(self, severity: str = "normal"):
        """
        severity: "normal" (mắt nhắm/gục đầu vừa vượt ngưỡng) hoặc
                  "escalated" (đã cảnh báo liên tục quá lâu mà chưa phản hồi)
        """
        now = time.time()

        if self._alert_start_time is None:
            self._alert_start_time = now

        elapsed = now - self._alert_start_time
        if elapsed > ALERT_ESCALATE_AFTER_SEC:
            severity = "escalated"

        if now - self._last_alert_time < ALERT_COOLDOWN_SEC:
            return  # đang trong thời gian cooldown, không phát chồng âm thanh

        self._last_alert_time = now

        if not self.enabled or self._sound is None:
            print(f"[ALERT:{severity.upper()}] (không có âm thanh - chỉ log)")
            return

        sound_to_play = self._escalated_sound if (severity == "escalated" and self._escalated_sound) else self._sound
        sound_to_play.play()

    def reset(self):
        """Gọi khi tài xế đã trở lại trạng thái tỉnh táo, để lần cảnh báo tiếp theo
        được tính lại từ đầu (không bị coi là 'escalated' ngay)."""
        self._alert_start_time = None
