"""Трекинг объектов: детекция YOLO + ByteTrack + сглаживание статуса."""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from .config import TrackerConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Отслеживаемый объект на кадре.

    Attributes:
        track_id: уникальный ID трека от трекера.
        box: bounding box [x1, y1, x2, y2].
        status: True — каска есть, False — каски нет,
                None — ещё не набрано достаточно кадров для решения.
    """
    track_id: int
    box: np.ndarray
    status: Optional[bool]


class StatusSmoother:
    """Сглаживает 'мерцание' класса голосованием по последним кадрам.

    Вынесена в отдельный класс, чтобы логику голосования можно было
    тестировать без загрузки YOLO-модели.
    """

    def __init__(self, history_len: int, min_hits: int, threshold: float):
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=history_len))
        self.min_hits = min_hits
        self.threshold = threshold

    def update(self, track_id: int, has_helmet: bool) -> Optional[bool]:
        """Добавляет наблюдение и возвращает итоговый статус (или None)."""
        self.history[track_id].append(int(has_helmet))
        hist = self.history[track_id]
        if len(hist) < self.min_hits:
            return None
        return (sum(hist) / len(hist)) >= self.threshold

    def forget(self, track_id: int) -> None:
        """Забывает трек (вызывается при cleanup)."""
        self.history.pop(track_id, None)


class HelmetTracker:
    """Трекает головы и каски, определяет статус защищённости каждого трека."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self.model = YOLO(config.model_path)
        self.smoother = StatusSmoother(
            history_len=config.history_len,
            min_hits=config.min_hits,
            threshold=config.voting_threshold,
        )
        self.last_seen: Dict[int, int] = {}
        self.frame_id = 0

    def update(self, frame: np.ndarray) -> List[Detection]:
        """Обрабатывает кадр и возвращает список отслеживаемых объектов."""
        self.frame_id += 1
        results = self.model.track(
            frame,
            conf=self.config.conf,
            iou=self.config.iou,
            classes=[self.config.head_id, self.config.helmet_id],
            persist=True,
            tracker=self.config.tracker,
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            self._cleanup()
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        detections: List[Detection] = []
        for k, (box, c) in enumerate(zip(xyxy, cls)):
            # Пропускаем объекты, которым трекер ещё не присвоил ID
            if ids is None or ids[k] < 0:
                continue

            track_id = int(ids[k])
            if not self._is_valid_box(box, frame):
                continue

            self.last_seen[track_id] = self.frame_id
            status = self.smoother.update(track_id, c == self.config.helmet_id)
            detections.append(Detection(track_id=track_id, box=box, status=status))

        self._cleanup()
        return detections

    def _is_valid_box(self, box: np.ndarray, frame: np.ndarray) -> bool:
        """Отсекает 'мусор': слишком маленькие или гигантские боксы."""
        w = box[2] - box[0]
        h = box[3] - box[1]
        if w < self.config.min_box_size or h < self.config.min_box_size:
            return False
        return w <= frame.shape[1] * self.config.max_box_ratio

    def _cleanup(self) -> None:
        """Удаляет треки, которые давно не встречались."""
        dead = [
            track_id for track_id, last in self.last_seen.items()
            if self.frame_id - last > self.config.max_lost
        ]
        for track_id in dead:
            self.smoother.forget(track_id)
            self.last_seen.pop(track_id, None)
            logger.debug("Удалён трек %d", track_id)