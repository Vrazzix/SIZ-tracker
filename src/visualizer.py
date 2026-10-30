"""Отрисовка результатов трекинга на кадре."""

from typing import List

import cv2
import numpy as np

from .tracker import Detection

COLOR_PENDING = (200, 200, 200)   # серый: статус ещё не определён
COLOR_HELMET = (0, 200, 0)        # зелёный: каска есть
COLOR_NO_HELMET = (0, 0, 255)     # красный: каски нет


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> None:
    """Рисует боксы и подписи статусов на кадре (in-place)."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det.box)

        if det.status is None:
            color, label = COLOR_PENDING, f"ID {det.track_id}: ..."
        elif det.status:
            color, label = COLOR_HELMET, f"ID {det.track_id}: HELMET"
        else:
            color, label = COLOR_NO_HELMET, f"ID {det.track_id}: NO HELMET"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)