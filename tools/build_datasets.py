"""Собирает датасеты для двух стратегий из V2 + V6 (все symlink, без копий).

Источники (проверено по сеткам + количественно):
  V2 (hard_hat_dataset):       0 = голое лицо без каски, 1 = человек в каске/каска (всё вместе)
  V6 (v2_github/.../v6):       0 = голова человека в каске, 1 = сама каска
  V6 не размечает голое лицо (260/262 бокса V2-0 отсутствуют в V6).

Итоговые классы:
  head   = V6-0 + V2-0  (каждая голова, в каске или без)
  helmet = V6-1         (сами каски, включая снятые/брошенные)

Вариант 3 (одна модель, 2 класса): 0=head, 1=helmet  -> datasets/ds_head_helmet
Вариант 2 (две модели):            datasets/ds_head (0=head), datasets/ds_helmet (0=helmet)

Разбивка train/val = каноническая V2 (4000/1000).
"""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = f'{BASE}/hard_hat_dataset'
V6 = f'{BASE}/v2_github/SIZ-tracker/hard_hat_dataset_v6'
OUT = f'{BASE}/datasets'


def read_labels(dsdir, split, name):
    p = f'{dsdir}/labels/{split}/{name}.txt'
    c0, c1 = [], []
    if os.path.exists(p):
        for line in open(p):
            s = line.split()
            if len(s) == 5:
                (c0 if s[0] == '0' else c1).append(s[1:5])
    return c0, c1


def image_path(imgdir, split, name):
    for ext in ('.png', '.jpg', '.jpeg', '.JPG', '.bmp'):
        p = f'{imgdir}/images/{split}/{name}{ext}'
        if os.path.exists(p):
            return p
    return None


def main():
    v2_train = sorted(f[:-4] for f in os.listdir(f'{V2}/labels/train') if f.endswith('.txt'))
    v2_val = sorted(f[:-4] for f in os.listdir(f'{V2}/labels/val') if f.endswith('.txt'))
    print(f'V2 split: train={len(v2_train)} val={len(v2_val)}')

    variants = {
        # имя: [(используемые боксы, class_id)] — head из (v6_0+v2_0), helmet из (v6_1)
        'ds_head_helmet': {'classes': 'both',  'names': ['head', 'helmet']},
        'ds_head':        {'classes': 'head',  'names': ['head']},
        'ds_helmet':      {'classes': 'helmet','names': ['helmet']},
    }
    for v in variants:
        for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
            os.makedirs(f'{OUT}/{v}/{sub}', exist_ok=True)

    stats = {v: dict(imgs=0, head_boxes=0, helmet_boxes=0, empty=0) for v in variants}

    for split, names in (('train', v2_train), ('val', v2_val)):
        for name in names:
            v6_0, v6_1 = read_labels(V6, split, name)
            v2_0, _ = read_labels(V2, split, name)
            heads = v6_0 + v2_0
            helmets = v6_1

            src = image_path(V2, split, name)
            if src is None:
                print('WARN image not found:', split, name)
                continue

            def emit(v, lines):
                p = f'{OUT}/{v}/labels/{split}/{name}.txt'
                with open(p, 'w') as f:
                    f.write('\n'.join(lines))
                st = stats[v]
                st['imgs'] += 1
                if not lines:
                    st['empty'] += 1
                # symlink картинки
                ip = f'{OUT}/{v}/images/{split}/{name}.png'
                if not os.path.exists(ip):
                    os.symlink(os.path.abspath(src), ip)

            for v, cfg in variants.items():
                if cfg['classes'] == 'both':
                    lines = [f'0 {b[0]} {b[1]} {b[2]} {b[3]}' for b in heads] + \
                            [f'1 {b[0]} {b[1]} {b[2]} {b[3]}' for b in helmets]
                elif cfg['classes'] == 'head':
                    lines = [f'0 {b[0]} {b[1]} {b[2]} {b[3]}' for b in heads]
                else:
                    lines = [f'0 {b[0]} {b[1]} {b[2]} {b[3]}' for b in helmets]
                emit(v, lines)
                if cfg['classes'] in ('both', 'head'):
                    stats[v]['head_boxes'] += len(heads)
                if cfg['classes'] in ('both', 'helmet'):
                    stats[v]['helmet_boxes'] += len(helmets)

    for v, cfg in variants.items():
        y = (f'path: {OUT}/{v}\ntrain: images/train\nval: images/val\n'
             f'nc: {len(cfg["names"])}\nnames:\n' +
             ''.join(f'- {n}\n' for n in cfg['names']))
        open(f'{OUT}/{v}/dataset.yaml', 'w').write(y)

    print()
    for v, s in stats.items():
        print(f'{v}: imgs={s["imgs"]} head_boxes={s["head_boxes"]} '
              f'helmet_boxes={s["helmet_boxes"]} empty_files={s["empty"]}')
    print('OK ->', OUT)


if __name__ == '__main__':
    main()
