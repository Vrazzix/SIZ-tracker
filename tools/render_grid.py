import os, glob, sys
import cv2

BASE = '/home/user/Shcheglov/Shcheglov/Helmets_and_masks'
COL = {0: (0, 200, 0), 1: (0, 80, 255), 2: (255, 160, 0)}  # класс -> цвет BGR


def parse(p):
    out = []
    if os.path.exists(p):
        for l in open(p):
            s = l.split()
            if len(s) == 5:
                out.append((int(s[0]) + 1, tuple(map(float, s[1:5]))))  # +1 => класс 1,2 (виднее)
    return out


def render(ds, labeldir, imgdir, bases, cell=300):
    tiles = []
    for base in bases:
        ip = None
        for ext in ('.png', '.jpg', '.jpeg', '.JPG'):
            p = f'{imgdir}/{base}{ext}'
            if os.path.exists(p):
                ip = p
                break
        if not ip:
            img = None
        else:
            img = cv2.imread(ip)
        if img is None:
            tiles.append(None)
            continue
        h, w = img.shape[:2]
        boxes = parse(f'{labeldir}/{base}.txt')
        for c, (cx, cy, bw, bh) in boxes:
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COL[c - 1], 2)
            cv2.putText(img, f'{c}', (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL[c - 1], 2)
        # подпись имени
        cv2.putText(img, base[:24], (4, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        img = cv2.resize(img, (cell, cell))
        tiles.append(img)
    good = [t for t in tiles if t is not None]
    if not good:
        return None
    n = len(good)
    cols = 3
    rows = (n + cols - 1) // cols
    grid = None
    for i in range(rows):
        rowtiles = []
        for j in range(cols):
            k = i * cols + j
            t = good[k] if k < n else None
            if t is None:
                t = ((255, 255, 255) if i % 2 else (230, 230, 230))
                t = cv2.UMat(t, dtype=cv2.CV_8UC3).get().reshape(cell, cell, 3) if isinstance(t, tuple) else \
                    __import__('numpy').full((cell, cell, 3), t, dtype='uint8')
            rowtiles.append(t)
        row = __import__('numpy').hstack(rowtiles)
        grid = row if grid is None else __import__('numpy').vstack((grid, row))
    return grid


# базовые картинки, где в v2 есть ОБА класса (человек в каске + голоший)
def pick_both(v2lab, n=9):
    out = []
    for f in sorted(os.listdir(v2lab)):
        if f == '.DS_Store':
            continue
        ls = [l.split() for l in open(f'{v2lab}/{f}') if l.strip()]
        if any(s[0] == '0' for s in ls) and any(s[0] == '1' for s in ls):
            out.append(f[:-4])
        if len(out) >= n:
            break
    return out


def pick_only(v2lab, cls, n=6):
    out = []
    for f in sorted(os.listdir(v2lab)):
        if f == '.DS_Store':
            continue
        ls = [l.split() for l in open(f'{v2lab}/{f}') if l.strip()]
        if ls and all(s[0] == cls for s in ls):
            out.append(f[:-4])
        if len(out) >= n:
            break
    return out


v2lab = f'{BASE}/hard_hat_dataset/labels/train'
v6lab = f'{BASE}/v2_github/SIZ-tracker/hard_hat_dataset_v6/labels/train'
v2img = f'{BASE}/hard_hat_dataset/images/train'
v6img = f'{BASE}/v2_github/SIZ-tracker/hard_hat_dataset_v6/images/train'

# общий набор баз (взять из v2, где есть оба класса)
both = pick_both(v2lab, 9)
g2 = render('v2', v2lab, v2img, both)
g6 = render('v6', v6lab, v6img, both)

if g2 is not None:
    cv2.putText(g2, 'V2  (green=head/0, orange=helmet/1)', (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.imwrite(f'{BASE}/debug_grid_v2.jpg', g2)
if g6 is not None:
    cv2.putText(g6, 'V6 (green=head_with_helmet/0, orange=helmet/1)', (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.imwrite(f'{BASE}/debug_grid_v6.jpg', g6)
print('v2 grid:', g2.shape if g2 is not None else None)
print('v6 grid:', g6.shape if g6 is not None else None)
print('bases:', both)
