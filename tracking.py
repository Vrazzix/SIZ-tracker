import cv2
from collections import defaultdict, deque
from ultralytics import YOLO

# ID классов в вашей модели
HEAD_ID, HELMET_ID = 0, 1

class HelmetTracker:
    def __init__(self, model_path, conf=0.5, iou=0.5,
                 history_len=15, min_hits=3, max_lost=30):
        """
        Трекает каски и головы без привязки друг к другу.
        Каска (helmet) = человек защищен.
        Голова (head) = человек без каски.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.history_len = history_len
        self.min_hits = min_hits
        self.max_lost = max_lost
        
        # История для сглаживания мерцания статуса (чтобы класс не прыгал)
        self.history = defaultdict(lambda: deque(maxlen=history_len))
        self.last_seen = {}
        self.frame_id = 0

    def update(self, frame):
        self.frame_id += 1
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=[HEAD_ID, HELMET_ID],
            persist=True,
            tracker='bytetrack.yaml',
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            self._cleanup()
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        detections = []  # (id, box, status)
        
        for k, (box, c) in enumerate(zip(xyxy, cls)):
            # Пропускаем объекты, которым трекер еще не присвоил ID
            if ids is None or ids[k] < 0:
                continue
                
            obj_id = int(ids[k])
            
            # Фильтр: отсекаем слишком маленькие или гигантские объекты (защита от мусора)
            w = box[2] - box[0]
            h = box[3] - box[1]
            if w < 15 or h < 15 or w > frame.shape[1] * 0.8:
                continue
                
            self.last_seen[obj_id] = self.frame_id
            
            # Определяем статус (1 - есть каска, 0 - нет каски)
            status_val = 1 if c == HELMET_ID else 0
            self.history[obj_id].append(status_val)
            
            h_hist = self.history[obj_id]
            
            # Если объект еще слишком "молодой" в трекере, не ставим окончательный статус
            if len(h_hist) < self.min_hits:
                final_status = None
            else:
                # Голосование: если в 70% кадров была каска, значит она есть
                final_status = (sum(h_hist) / len(h_hist)) >= 0.6
                
            detections.append((obj_id, box, final_status))

        self._cleanup()
        return detections

    def _cleanup(self):
        # Удаляем треки, которые давно не видели
        dead = [k for k, v in self.last_seen.items()
                if self.frame_id - v > self.max_lost]
        for k in dead:
            self.history.pop(k, None)
            self.last_seen.pop(k, None)


def draw(frame, detections):
    for obj_id, box, status in detections:
        x1, y1, x2, y2 = map(int, box)
        
        if status is None:
            # Еще не набрали статистику (первые кадры появления)
            color, label = (200, 200, 200), f'ID {obj_id}: ...'
        elif status:
            # Модель нашла каску
            color, label = (0, 200, 0),   f'ID {obj_id}: HELMET'
        else:
            # Модель нашла голову (без каски)
            color, label = (0, 0, 255),   f'ID {obj_id}: NO HELMET'
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# ---------- НАСТРОЙКИ ----------
MODEL_PATH = '/home/user/Shcheglov/Shcheglov/Helmets_and_masks/runs/detect/hard_hat_yolov26n_no_person/weights/best.pt'
VIDEO_IN   = '/home/user/Shcheglov/Shcheglov/Helmets_and_masks/Importance-of-wearing-Safety-helmets-at-work-_IsNo6F3gNHw_.mp4'
VIDEO_OUT  = 'results_helmets_out.mp4'

# Инициализация (conf=0.5 чтобы отсечь ложные каски на фоне)
tracker = HelmetTracker(MODEL_PATH, conf=0.5, iou=0.6)

# ---------- ОСНОВНОЙ ЦИКЛ ----------
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    print(f"Ошибка: Не удалось открыть видео {VIDEO_IN}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:
    fps = 25.0

ret, first_frame = cap.read()
if not ret:
    print("Ошибка: Не удалось прочитать первый кадр.")
    exit()

h, w = first_frame.shape[:2]

# Пытаемся использовать хороший кодек, откатываемся на mp4v если не вышло
fourcc = cv2.VideoWriter_fourcc(*'avc1')
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

if not writer.isOpened():
    print("Кодек avc1 не поддерживается, пробуем mp4v...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

if not writer.isOpened():
    print("Ошибка: Не удалось инициализировать VideoWriter.")
    exit()

frame_count = 0

try:
    while True:
        if frame_count == 0:
            frame = first_frame
        else:
            ret, frame = cap.read()
            if not ret:
                break

        detections = tracker.update(frame)
        draw(frame, detections)
        
        writer.write(frame)
            
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Обработано кадров: {frame_count}")

except Exception as e:
    print(f"❌ Произошла ошибка во время обработки: {e}")

finally:
    print("Завершаем запись видео...")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"✅ Готово! Видео сохранено в {VIDEO_OUT}")