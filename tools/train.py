"""Обучение моделей head/helmet с логированием в MLflow.

Запуск:
  .venv-train/bin/python tools/train.py --tag v3 --data datasets/ds_head_helmet/dataset.yaml
  .venv-train/bin/python tools/train.py --tag v2-head --data datasets/ds_head/dataset.yaml
  .venv-train/bin/python tools/train.py --tag v2-helmet --data datasets/ds_helmet/dataset.yaml

MLflow: локальное file-хранилище mlruns/, эксперимент HelmetsAndMasks,
tag в названии run. Метрики: loss'ы + P/R/mAP по эпохам, артефакты: best.pt, results.csv.
"""
import argparse
import csv
import os
import sys
import time

import mlflow
from ultralytics import YOLO

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='метка эксперимента (v3, v2-head, ...)')
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', default=f'{BASE}/yolo26n.pt')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--resume-from', default=None, help='weights dir для resume (runs/.../name)')
    ap.add_argument('--project', default=f'{BASE}/runs/mlflow_train')
    args = ap.parse_args()

    os.environ.setdefault('MLFLOW_ARTIFACTS_URI', os.path.join(BASE, 'mlflow_artifacts'))
    mlflow.set_tracking_uri('sqlite:///' + os.path.join(BASE, 'mlflow.db'))
    mlflow.set_experiment('HelmetsAndMasks')

    names = {
        'ds_head_helmet': 'v3 one model: head+helmet',
        'ds_head': 'v2 head detector',
        'ds_helmet': 'v2 helmet detector',
    }
    run_name = f'{args.tag} ({os.path.basename(os.path.dirname(args.data))})'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'tag': args.tag,
            'model': args.model,
            'data': os.path.abspath(args.data),
            'epochs': args.epochs,
            'patience': args.patience,
            'batch': args.batch,
            'imgsz': args.imgsz,
            'torch': __import__('torch').__version__,
            'ultralytics': __import__('ultralytics').__version__,
            'gpu': __import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else 'cpu',
            'description': names.get(os.path.basename(os.path.dirname(args.data)), ''),
        })

        start = time.time()
        if args.resume_from:
            model = YOLO(os.path.join(args.resume_from, 'weights', 'last.pt'))
            r = model.train(resume=True)
        else:
            model = YOLO(args.model)
            r = model.train(
                data=args.data,
                epochs=args.epochs,
                patience=args.patience,
                batch=args.batch,
                imgsz=args.imgsz,
                device=0,
                project=args.project,
                name=args.tag,
                workers=4,
                exist_ok=True,
                plots=False,
            )
        dur = time.time() - start

        # финальные метрики val
        best = os.path.join(r.save_dir, 'weights', 'best.pt')
        if os.path.exists(best):
            res = model.val(data=args.data, plots=False, verbose=False, device=0)
            box = res.box  # DetMetrics
            step = int(getattr(res, 'epoch', args.epochs))
            mp, mr = float(box.mp), float(box.mr)
            ap50_all, ap_all = float(box.ap50), float(box.ap)
            mlflow.log_metric('val/precision_all', mp, step=step)
            mlflow.log_metric('val/recall_all', mr, step=step)
            mlflow.log_metric('val/map50_all', ap50_all, step=step)
            mlflow.log_metric('val/map50_95_all', ap_all, step=step)
            for i, nm in enumerate(res.names.values()):
                m = box.metrics[0]
                try:
                    mlflow.log_metric(f'val/precision_{nm}', float(m[0][i]), step=step)
                    mlflow.log_metric(f'val/recall_{nm}', float(m[1][i]), step=step)
                    mlflow.log_metric(f'val/map50_{nm}', float(m[4][i]), step=step)
                    mlflow.log_metric(f'val/map50_95_{nm}', float(m[3][i]), step=step)
                except (IndexError, TypeError):
                    pass
            mlflow.log_metric('train/duration_min', round(dur / 60, 1))

        # loss-кривые по эпохам
        csv_path = os.path.join(r.save_dir, 'results.csv')
        if os.path.exists(csv_path):
            import re
            def _clean(name):
                # mlflow: только alnum _ - . space : /
                return re.sub(r'[^A-Za-z0-9_\-\. :/]', '', name)
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            epoch_col = rows[0].keys().__iter__().__next__()
            for row in rows:
                step = int(row.get(epoch_col) or row.get('epoch') or 0)
                for k, v in row.items():
                    if k == epoch_col:
                        continue
                    try:
                        mlflow.log_metric(f'loss/{_clean(k)}', float(v), step=step)
                    except (TypeError, ValueError):
                        pass
            mlflow.log_text(open(csv_path).read()[:20000], 'results_csv_head.txt')

        # артефакты
        for p in ('weights/best.pt', 'weights/last.pt', 'args.yaml'):
            fp = os.path.join(r.save_dir, p)
            if os.path.exists(fp):
                mlflow.log_artifact(fp)
        print(f'MLFLOW RUN {run_name} DONE -> {r.save_dir}')


if __name__ == '__main__':
    main()
