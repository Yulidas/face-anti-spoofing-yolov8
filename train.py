#!/usr/bin/env python3
"""Fine-tune a YOLOv8 detector for face presentation attack detection.

Provides a command-line training entry point with configurable pretrained
weights, dataset YAML, image resolution, epochs, batch size, and device.
Supports separate experiments on standard-color and HSV image datasets.
"""
import argparse
from pathlib import Path


def cli():
    p = argparse.ArgumentParser(description='Train a YOLOv8 anti-spoofing model')
    p.add_argument('--model', required=True, help='Pretrained YOLOv8 model OR existing .pt checkpoint')
    p.add_argument('--data', type=Path, default=Path('dataset/SplitData/data.yaml'))
    p.add_argument('--epochs', type=int, default=160)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=-1)
    p.add_argument('--resume', action='store_true', help='Resume original Ultralytics training session')
    p.add_argument('--device', default=None, help='e.g. cpu, 0')
    return p.parse_args()


def main():
    args = cli()
    if not args.data.is_file():
        raise SystemExit(f'Dataset configuration missing: {args.data}. The private dataset is not shipped.')
    from ultralytics import YOLO
    model = YOLO(args.model)
    options = dict(data=str(args.data.resolve()), imgsz=args.imgsz, epochs=args.epochs,
                   batch=args.batch, patience=0, cache=False)
    if args.device:
        options['device'] = args.device
    if args.resume:
        options['resume'] = True
    model.train(**options)


if __name__ == '__main__':
    main()
