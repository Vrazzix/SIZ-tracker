"""Сетка-превью датасета реальных голов: что получит модель при дообучении.
Зелёный = head (верх person), оранжевый = helmet."""
import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = f'{BASE}/real_world_frames'
COL = {0: (0, 200, 0), 1: (0, 80, 255)}


def boxes_of(p):
    out = []
    if os.path.exists(p):
        for l in open(p):
            s = l.split()
            if len(s) == 5:
                out.append((int(s[0]), tuple(map(float, s[1:5]))))
    return out


def pick(split, n):
    d = f'{DS}/{split}/images'
    if not os.path.isdir(d):
        return []
    names = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.jpg'))
    # предпочитаем кадры, где есть оба класса
    def has_both(nm):
        bs = boxes_of(f'{DS}/{split}/labels/{nm}.txt')
        return any(c == 0 for c, _ in bs) and any(c == 1 for c, _ in bs)
    both = [x for x in names if has_both(x)]
    rest = [x for x in names if not has_both(x)]
    chosen = (both + rest)[:n]
    return chosen


def render(names, split, cell=360):
    tiles = []
    for name in names:
        img = cv2.imread(f'{DS}/{split}/images/{name}.jpg')
        if img is None:
            continue
        h, w = img.shape[:2]
        for c, (cx, cy, bw, bh) in boxes_of(f'{DS}/{split}/labels/{name}.txt'):
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COL[c], 2)
            cv2.putText(img, str(c), (x1, max(y1 - 5, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[c], 2)
        cv2.putText(img, name[:30], (4, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (cell, cell)))
    if not tiles:
        return None
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    grid = None
    for i in range(rows):
        rowt = []
        for j in range(cols):
            k = i * cols + j
            rowt.append(tiles[k] if k < len(tiles) else np.full((cell, cell, 3), 235, dtype='uint8'))
        row = np.hstack(rowt)
        grid = row if grid is None else np.vstack((grid, row))
    return grid


mixed = pick('train', 6) + pick('val', 3)
g = render(mixed, 'train') if mixed else None
if g is None and mixed:
    # если имена перемешаны train/val — рендерим поочерёдно
    pass
# проще: рендер по списку с поиском сплита
tiles = []
for name in mixed:
    split = None
    for s in ('train', 'val'):
        if os.path.exists(f'{DS}/{s}/images/{name}.jpg'):
            split = s
            break
    img = cv2.imread(f'{DS}/{split}/images/{name}.jpg')
    h, w = img.shape[:2]
    for c, (cx, cy, bw, bh) in boxes_of(f'{DS}/{split}/labels/{name}.txt'):
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), COL[c], 2)
        cv2.putText(img, str(c), (x1, max(y1 - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[c], 2)
    cv2.putText(img, name[:30], (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    tiles.append(cv2.resize(img, (360, 360)))
if tiles:
    cols = 3
    grid = None
    for i in range((len(tiles) + cols - 1) // cols):
        rowt = []
        for j in range(cols):
            k = i * cols + j
            rowt.append(tiles[k] if k < len(tiles) else np.full((360, 360, 3), 235, dtype='uint8'))
        row = np.hstack(rowt)
        grid = row if grid is None else np.vstack((grid, row))
    cv2.putText(grid, 'real-world heads: green=head(0) orange=helmet(1)', (6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    out = f'{BASE}/debug/real_heads_preview.jpg'
    cv2.imwrite(out, grid)
    print('grid ->', out, grid.shape)
    print('кадры:', mixed)
