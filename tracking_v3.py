"""V3 — одна модель (head + helmet), матчинг head<->helmet, stray-каска.

Статусы:
  PROTECTED   — head с helmet (каска над головой)
  NO_HELMET   — head без helmet
  STRAY       — helmet без head (снята/брошена)
"""
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

HEAD_ID, HELMET_ID = 0, 1
# helmet должен "сидеть" в расширенной области головы (сверху) — порог:
# центр каски в расширенном bbox головы, либо IoU > threshold
MATCH_IOU = 0.12
MATCH_CENTER_Y = 1.6  # центр каски может быть выше центра головы до 1.6x высоты головы

# Сглаживание статуса (голосование по истории)
HISTORY_LEN = 12
MIN_HITS = 3
THRESHOLD = 0.6

class V3Tracker:
    def __init__(self, model_path, conf=0.35, iou=0.5, min_size=15, max_w_ratio=0.8):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.min_size = min_size
        self.max_w_ratio = max_w_ratio
        self.frame_id = 0
        # история: {track_id: deque[status_int]} — 1=helmet на этой голове
        self.history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
        self.last_seen = {}
        # stray-треки отдельно
        self.stray_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
        self.stray_last = {}

    @staticmethod
    def _valid(box, frame_w):
        w = box[2] - box[0]; h = box[3] - box[1]
        return w >= 15 and h >= 15 and w <= frame_w * 0.8

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1: return 0.0
        inter = (x2 - x1) * (y2 - y1)
        aa = (a[2]-a[0]) * (a[3]-a[1]); bb = (b[2]-b[0]) * (b[3]-b[1])
        return inter / max(aa * bb, 1e-9)

    @staticmethod
    def _helmet_on_head(helmet, head):
        # центр каски внутри расширенной головы (вверх)
        hx1, hy1, hx2, hy2 = helmet
        cx = (hx1 + hx2) / 2; cy = (hy1 + hy2) / 2
        hcx = (head[0] + head[2]) / 2
        hcy = (head[1] + head[3]) / 2
        hh = head[3] - head[1]
        hw = head[2] - head[0]
        # каска должна быть в пределах ширины головы (с запасом) и не сильно ниже
        if abs(cx - hcx) > hw * 0.75: return False
        if cy > (head[3] + hh * 0.25): return False          # не ниже нижней кромки головы + 25%
        if cy < (head[1] - hh * (MATCH_CENTER_Y - 0.5)): return False  # не слишком высоко
        # ИЛИ достаточное перекрытие
        return True

    def update(self, frame):
        self.frame_id += 1
        h, w = frame.shape[:2]
        res = self.model.track(frame, conf=self.conf, iou=self.iou, persist=True,
                               tracker='bytetrack.yaml', verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return [], []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        heads = []   # (id, box)
        helmets = [] # (id, box)
        for k in range(len(xyxy)):
            if ids is not None and ids[k] < 0: continue
            oid = int(ids[k]) if ids is not None else k  # fallback: индекс (если трекер не дал ID)
            box = xyxy[k]
            if not self._valid(box, w): continue
            if int(cls[k]) == HEAD_ID: heads.append((oid, box))
            elif int(cls[k]) == HELMET_ID: helmets.append((oid, box))

        # Матчинг: каждая каска -> ближайшая подходящая голова (greedy по IoU/fit)
        pairs = []  # (helmet_id, head_id, score)
        for hid, hbox in heads:
            for lid, lbox in helmets:
                if self._helmet_on_head(lbox, hbox) or self._iou(lbox, hbox) >= MATCH_IOU:
                    score = self._iou(lbox, hbox) + (1.0 if self._helmet_on_head(lbox, hbox) else 0.0)
                    pairs.append((score, hid, lid))
        pairs.sort(reverse=True)
        matched_heads = set(); matched_helmets = set()
        head_has_helmet = {}
        for score, hid, lid in pairs:
            if hid in matched_heads or lid in matched_helmets: continue
            matched_heads.add(hid); matched_helmets.add(lid)
            head_has_helmet[hid] = True

        # Статусы с голосованием
        def vote(store, tid, value):
            d = store[tid]; d.append(int(value))
            self._last_seen(store, tid)
            if len(d) < MIN_HITS: return None
            return (sum(d) / len(d)) >= THRESHOLD

        out = []
        for hid, hbox in heads:
            st = vote(self.history, hid, 1 if head_has_helmet.get(hid) else 0)
            out.append((hid, hbox, st))
        stray = []
        for lid, lbox in helmets:
            if lid in matched_helmets: continue
            st = vote(self.stray_history, lid, 1)
            stray.append((lid, lbox, st))

        self._cleanup()
        return out, stray

    def _last_seen(self, store, tid):
        self.last_seen[tid] = self.frame_id

    def _cleanup(self):
        dead = [k for k, v in self.last_seen.items() if self.frame_id - v > 40]
        for k in dead:
            self.history.pop(k, None)
            self.stray_history.pop(k, None)
            self.stray_last.pop(k, None)
            self.last_seen.pop(k, None)


def draw(frame, detections, stray):
    for tid, box, st in detections:
        x1, y1, x2, y2 = map(int, box)
        if st is None:
            color, label = (200, 200, 200), f'ID{tid}: ...'
        elif st:
            color, label = (0, 200, 0), f'ID{tid}: PROTECTED'
        else:
            color, label = (0, 0, 255), f'ID{tid}: NO HELMET'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    for tid, box, st in stray:
        x1, y1, x2, y2 = map(int, box)
        color = (0, 160, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'HELMET-{tid}: STRAY', (x1, max(y1 - 10, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


if __name__ == '__main__':
    MODEL_PATH = 'runs/mlflow_train/v3/weights/best.pt'
    VIDEO_IN = 'Запись с веб-камеры строительство дома из профилированного бруса [get-save.com].mp4'
    VIDEO_OUT = 'results_v3_out.mp4'

    tracker = V3Tracker(MODEL_PATH, conf=0.35, iou=0.5)

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise SystemExit(f'Не удалось открыть {VIDEO_IN}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps: fps = 25.0
    ret, first = cap.read()
    if not ret:
        raise SystemExit('Нет первого кадра')
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit('Не удалось открыть VideoWriter')

    n = 0
    try:
        while True:
            frame = first if n == 0 else cap.read()[1]
            if n > 0 and frame is None: break
            det, stray = tracker.update(frame)
            draw(frame, det, stray)
            writer.write(frame)
            n += 1
            if n % 100 == 0:
                print(f'frame {n}')
    finally:
        cap.release(); writer.release()
        print(f'GOTOWO: {n} frames -> {VIDEO_OUT}')
