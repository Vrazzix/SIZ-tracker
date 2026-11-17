"""B1: реальные головы из верха person-бокса (COCO yolo26n) + каски (v3).

Собирает датасет real_world_frames/:
  - кадры из видео (шаг STEP)
  - person (COCO) -> head = верх бокса человека
  - helmet (наша v3) -> класс helmet; если над head — пара head+helmet
  - stray helmet (без человека) -> только helmet

Классы: 0=head, 1=helmet (совместимо с v3).
Порог качества: head-высота >= MIN_HEAD_H px (иначе лицо слишком мелкое).
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3_PATH = f'{BASE}/runs/mlflow_train/v3/weights/best.pt'
COCO_PATH = f'{BASE}/yolo26n.pt'
OUT = f'{BASE}/real_world_frames'

VIDEOS = [
    f'{BASE}/Запись с веб-камеры строительство дома из профилированного бруса [get-save.com].mp4',
    f'{BASE}/Importance-of-wearing-Safety-helmets-at-work-_IsNo6F3gNHw_.mp4',
]
STEP = 5            # каждый N-й кадр
MIN_HEAD_H = 10     # мин. высота head-бокса в px
HEAD_FRAC = 0.30    # head = верхние 30% person-бокса
PERSON_CONF = 0.35
HELMET_CONF = 0.30
VAL_SPLIT = 0.10


def main():
    person = YOLO(COCO_PATH)
    v3 = YOLO(V3_PATH)

    all_frames = []  # (video_idx, frame_idx, path)
    vi = 0
    for video in VIDEOS:
        if not os.path.exists(video):
            print('нет видео:', video)
            continue
        cap = cv2.VideoCapture(video)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        while idx < n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                break
            # --- детекции ---
            rp = person.predict(f, conf=PERSON_CONF, classes=[0], verbose=False)[0]
            rh = v3.predict(f, conf=HELMET_CONF, verbose=False)[0]

            boxes_p = rp.boxes
            persons = []
            for i in range(len(boxes_p)):
                x1, y1, x2, y2 = boxes_p.xyxy[i].tolist()
                if (x2 - x1) < 15 or (y2 - y1) < 25:
                    continue
                persons.append((x1, y1, x2, y2))

            helmets = []
            for i in range(len(rh.boxes)):
                if int(rh.boxes.cls[i]) != 1:
                    continue
                x1, y1, x2, y2 = rh.boxes.xyxy[i].tolist()
                if (x2 - x1) < 8 or (y2 - y1) < 6:
                    continue
                helmets.append((x1, y1, x2, y2))

            if not persons and not helmets:
                idx += STEP
                continue

            # --- head из верха person ---
            labels = []
            H, W = f.shape[:2]
            heads_kept = 0
            for (x1, y1, x2, y2) in persons:
                bw = x2 - x1
                head_h = (y2 - y1) * HEAD_FRAC
                if head_h < MIN_HEAD_H:
                    continue
                hx1, hy1, hx2, hy2 = x1, y1, x2, y1 + head_h
                cx = (hx1 + hx2) / 2 / W
                cy = (hy1 + hy2) / 2 / H
                w = bw / W
                h = head_h / H
                labels.append([0, cx, cy, w, h])
                heads_kept += 1

            # --- каски ---
            kept_helmets = 0
            for (lx1, ly1, lx2, ly2) in helmets:
                labels.append([1, (lx1 + lx2) / 2 / W, (ly1 + ly2) / 2 / H,
                               (lx2 - lx1) / W, (ly2 - ly1) / H])
                kept_helmets += 1

            if not labels:
                idx += STEP
                continue

            # --- сохранить ---
            safe = f'v{vi}_f{idx:06d}'
            img_path = f'{OUT}/_all/images/{safe}.jpg'
            lab_path = f'{OUT}/_all/labels/{safe}.txt'
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            os.makedirs(os.path.dirname(lab_path), exist_ok=True)
            cv2.imwrite(img_path, f, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(lab_path, 'w') as lf:
                for c, cx, cy, w, h in labels:
                    lf.write(f'{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
            all_frames.append((safe, heads_kept, kept_helmets))
            idx += STEP
        cap.release()
        vi += 1

    # --- split train/val: раскидываем из _all/ ---
    import random
    import shutil
    random.seed(42)
    names = [x[0] for x in all_frames]
    random.shuffle(names)
    n_val = max(1, int(len(names) * VAL_SPLIT))
    val = set(names[:n_val])
    for kind, ext in (('images', '.jpg'), ('labels', '.txt')):
        srcdir = f'{OUT}/_all/{kind}'
        os.makedirs(f'{OUT}/train/{kind}', exist_ok=True)
        os.makedirs(f'{OUT}/val/{kind}', exist_ok=True)
        for safe in names:
            src = f'{srcdir}/{safe}{ext}'
            dst = f'{OUT}/{"val" if safe in val else "train"}/{kind}/{safe}{ext}'
            if os.path.exists(src):
                shutil.move(src, dst)
    shutil.rmtree(f'{OUT}/_all', ignore_errors=True)

    n_train = len([f for f in os.listdir(f'{OUT}/train/images') if f.endswith('.jpg')])
    n_val2 = len([f for f in os.listdir(f'{OUT}/val/images') if f.endswith('.jpg')])
    total_h = sum(x[1] for x in all_frames)
    total_l = sum(x[2] for x in all_frames)
    print(f'frames total={len(all_frames)} train={n_train} val={n_val2}')
    print(f'head labels={total_h}, helmet labels={total_l}')

    yaml = f'''# реальные кадры: 0=head (верх person-бокса), 1=helmet (v3)
path: {OUT}
train: train/images
val: val/images
nc: 2
names:
- head
- helmet
'''
    open(f'{OUT}/dataset.yaml', 'w').write(yaml)
    print('OK ->', OUT)


if __name__ == '__main__':
    main()
