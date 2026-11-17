"""Контрольная сетка по собранному датасету ds_head_helmet.
Зелёный = head(0), оранжевый = helmet(1)."""
import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = f'{BASE}/datasets/ds_head_helmet'
COL = {0: (0, 200, 0), 1: (0, 80, 255)}


def boxes_of(p):
    out = []
    if os.path.exists(p):
        for l in open(p):
            s = l.split()
            if len(s) == 5:
                out.append((int(s[0]), tuple(map(float, s[1:5]))))
    return out


def pick(n, need_stray=False):
    """n файлов train; если need_stray — только те, где есть helmet без head рядом."""
    names = []
    for f in sorted(os.listdir(f'{DS}/labels/train')):
        if not f.endswith('.txt'):
            continue
        bs = boxes_of(f'{DS}/labels/train/{f}')
        if need_stray:
            has_helmet = any(c == 1 for c, _ in bs)
            # "stray" = есть helmet, но мало head (или нет)
            n_head = sum(1 for c, _ in bs if c == 0)
            n_helm = sum(1 for c, _ in bs if c == 1)
            if has_helmet and n_head <= max(1, n_helm - 1):
                names.append(f[:-4])
        else:
            names.append(f[:-4])
        if len(names) >= n:
            break
    return names


def render(names, cell=300):
    tiles = []
    for name in names:
        img = cv2.imread(f'{DS}/images/train/{name}.png')
        if img is None:
            continue
        h, w = img.shape[:2]
        for c, (cx, cy, bw, bh) in boxes_of(f'{DS}/labels/train/{name}.txt'):
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COL[c], 2)
            cv2.putText(img, str(c), (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL[c], 2)
        cv2.putText(img, name[:26], (4, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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
            if k < len(tiles):
                rowt.append(tiles[k])
            else:
                rowt.append(np.full((cell, cell, 3), 235, dtype='uint8'))
        row = np.hstack(rowt)
        grid = row if grid is None else np.vstack((grid, row))
    return grid


mixed = pick(6)
stray = pick(4, need_stray=True)
g = render(mixed + stray)
if g is not None:
    cv2.putText(g, 'ds_head_helmet: green=head(0)  orange=helmet(1)  [6 mixed + 4 stray]',
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    out = f'{BASE}/debug_grid_ds.jpg'
    cv2.imwrite(out, g)
    print('grid:', g.shape, '->', out)
    print('mixed:', mixed)
    print('stray:', stray)
