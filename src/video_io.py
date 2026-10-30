"""Работа с видеофайлами: чтение и запись с fallback кодеков."""

import logging

import cv2

logger = logging.getLogger(__name__)


def open_reader(path: str) -> cv2.VideoCapture:
    """Открывает видео для чтения. Бросает исключение при ошибке."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    return cap


def read_fps(cap: cv2.VideoCapture, fallback: float) -> float:
    """Читает fps из файла; возвращает fallback, если значение 0 или NaN."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        logger.warning("Не удалось прочитать fps, использую %.1f", fallback)
        return fallback
    return fps


def create_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Создаёт VideoWriter, перебирая кодеки по порядку."""
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            logger.info("Использую кодек %s", codec)
            return writer
        logger.warning("Кодек %s не поддерживается, пробую следующий", codec)
    raise RuntimeError(f"Не удалось инициализировать VideoWriter: {path}")