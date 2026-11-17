# Детектор средств индивидуальной защиты на сотрудниках предприятия

Определение «человек в каске / без каски / каска снята (stray)» по видео с
видеонаблюдения. YOLO + Bytetrack.

## Состав

| Файл | Назначение |
|---|---|
| `tracking_v3.py` | Основной скрипт: одна модель (head + helmet), матчинг каска→голова, stray-каска, запись результата в mp4 |
| `tracking.py` | Старый скрипт (head_with_helmet/helmet) — для справки |
| `tools/build_datasets.py` | Сборка датасетов head/helmet из v2 + v6 разметок (symlink, без копий) |
| `tools/train.py` | Обучение с логированием в MLflow (sqlite: `mlflow.db`) |
| `tools/render_grid.py`, `tools/render_ds.py` | Отладочные сетки размеченных кадров |
| `hard_hat_dataset/` | v2 разметка: 0=head (без каски), 1=helmet (images/ — локально, не в git) |
| `hard_hat_dataset_v6/` | v6 разметка: 0=head_with_helmet, 1=helmet (images/ — локально, не в git) |
| `weights/v3_head_helmet_best.pt` | Обученная v3-модель (head+helmet, 100 эпох) |
| `yolo26n.pt` | Базовая модель для дообучения |
| `yolov8n-face-lindevs.pt` | Face-модель (для диагностики домен-разрыва) |
| `load_and_preprocess.ipynb` | Препроцессинг исходных XML-разметок |

## Как запустить

```bash
uv venv --python 3.12 .venv-train
uv pip install -r pyproject.toml          # torch (cu12x), ultralytics, mlflow
python tracking_v3.py                     # прогон по видео, результат -> results_v3_out.mp4
python tools/train.py --tag v3 --data <датасет>/dataset.yaml   # обучение + mlflow
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Стратегии сравнения

- **v3** — одна модель с классами `head` + `helmet`, решение и stray-логика в коде.
- **v2** — две модели (детектор голов + детектор касок), матчинг по ROI.

Вывод по v3 на реальном видео: каски детектит, головы — плохо (мелкие вдали,
низкое качество веб-камеры; предобученная face-модель на тех же кадрах также
находит 0–1 лиц). Улучшение: добавить в датасет реальные мелкие головы со
видео стройки.

## Демонстрация

<video src="https://github.com/user-attachments/assets/85929d5a-6741-4925-8d4d-e07bb1c564e1" controls width="640"></video>
