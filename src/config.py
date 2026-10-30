"""Конфигурация проекта: dataclass-и и загрузка из YAML."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml


@dataclass
class TrackerConfig:
    """Параметры детектора и трекера."""
    model_path: str
    conf: float = 0.5               # порог уверенности детекции
    iou: float = 0.6                # порог IoU для ассоциации треков
    history_len: int = 15           # окно истории статусов (кадров)
    min_hits: int = 3               # минимум кадров до финального статуса
    max_lost: int = 30              # сколько кадров хранить потерянный трек
    voting_threshold: float = 0.6   # доля кадров с каской для статуса True
    head_id: int = 0                # ID класса "голова"
    helmet_id: int = 1              # ID класса "каска"
    tracker: str = "bytetrack.yaml"
    min_box_size: int = 15          # фильтр мусора: мин. размер бокса (px)
    max_box_ratio: float = 0.8      # фильтр мусора: макс. доля ширины кадра


@dataclass
class VideoConfig:
    """Параметры входного и выходного видео."""
    input_path: str
    output_path: str = "results_helmets_out.mp4"
    fallback_fps: float = 25.0      # fps, если из файла прочитать не удалось


@dataclass
class AppConfig:
    """Корневая конфигурация приложения."""
    tracker: TrackerConfig
    video: VideoConfig

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        try:
            tracker = TrackerConfig(**raw.get("tracker", {}))
            video = VideoConfig(**raw.get("video", {}))
        except TypeError as e:
            raise ValueError(f"Ошибка в конфиге: {e}") from e
        return cls(tracker=tracker, video=video)