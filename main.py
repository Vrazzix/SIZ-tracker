"""Точка входа: пайплайн обработки видео."""

import argparse
import logging

from src.config import AppConfig
from src.tracker import HelmetTracker
from src.video_io import create_writer, open_reader, read_fps
from src.visualizer import draw_detections

logger = logging.getLogger("siz_tracker")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Детекция и трекинг касок на видео")
    parser.add_argument("--config", default="config.yaml", help="Путь к YAML-конфигу")
    parser.add_argument("--video", help="Переопределить путь к входному видео")
    parser.add_argument("--model", help="Переопределить путь к модели")
    parser.add_argument("--output", help="Переопределить путь к выходному видео")
    return parser.parse_args()


def process_video(config: AppConfig) -> None:
    """Основной пайплайн: чтение → трекинг → отрисовка → запись."""
    tracker = HelmetTracker(config.tracker)

    cap = open_reader(config.video.input_path)
    fps = read_fps(cap, config.video.fallback_fps)

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Не удалось прочитать первый кадр: {config.video.input_path}")
    h, w = first_frame.shape[:2]

    writer = create_writer(config.video.output_path, fps, w, h)

    frame_count = 0
    frame = first_frame
    try:
        while frame is not None:
            detections = tracker.update(frame)
            draw_detections(frame, detections)
            writer.write(frame)

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info("Обработано кадров: %d", frame_count)

            ok, frame = cap.read()
            if not ok:
                frame = None
    finally:
        cap.release()
        writer.release()
        logger.info("Готово: %d кадров сохранено в %s",
                    frame_count, config.video.output_path)


def main() -> None:
    setup_logging()
    args = parse_args()

    config = AppConfig.from_yaml(args.config)
    # CLI-аргументы переопределяют конфиг
    if args.video:
        config.video.input_path = args.video
    if args.model:
        config.tracker.model_path = args.model
    if args.output:
        config.video.output_path = args.output

    try:
        process_video(config)
    except RuntimeError as e:
        logger.error("%s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()