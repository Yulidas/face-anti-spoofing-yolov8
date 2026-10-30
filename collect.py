#!/usr/bin/env python3
"""Face image collection and automatic YOLO annotation.

Captures frames from a webcam or processes an existing image directory,
locates faces, filters unsuitable detections, and saves bounding-box
annotations in YOLO format. Ground-truth class labels are supplied by
the operator; this module does not establish image authenticity.
"""
import argparse
from pathlib import Path
from time import time_ns


def collect_one(frame, detector, class_id, min_score, blur_threshold):
    import cv2
    preview = frame.copy()
    _, boxes = detector.findFaces(frame.copy(), draw=False)
    h_img, w_img = frame.shape[:2]
    labels = []
    for bbox in boxes or []:
        if float(bbox['score'][0]) < min_score:
            continue
        x, y, w, h = [int(v) for v in bbox['bbox']]
        # Original margin convention: 10% horizontal and asymmetric vertical margin
        offset_w, offset_h = int(w * 0.1), int(h * 0.2)
        x1, y1 = max(0, x - offset_w), max(0, y - int(2.5 * offset_h))
        x2, y2 = min(w_img, x + w + offset_w), min(h_img, y + h + int(offset_h))
        if x2 <= x1 or y2 <= y1:
            continue
        face = frame[y1:y2, x1:x2]
        if face.size == 0 or cv2.Laplacian(face, cv2.CV_64F).var() <= blur_threshold:
            continue
        cx, cy = (x1 + x2) / (2 * w_img), (y1 + y2) / (2 * h_img)
        nw, nh = (x2 - x1) / w_img, (y2 - y1) / h_img
        labels.append(f'{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return labels, preview


def main():
    p = argparse.ArgumentParser(description='Collect YOLO-labeled faces; dataset is not published')
    p.add_argument('--source', default='0', help='Webcam index or folder of images')
    p.add_argument('--class-id', type=int, choices=[0, 1], required=True, help='0=Fake; 1=Real')
    p.add_argument('--output', type=Path, default=Path('dataset/All'))
    p.add_argument('--min-score', type=float, default=0.8)
    p.add_argument('--blur-threshold', type=float, default=35.)
    args = p.parse_args()
    try:
        import cv2
        from cvzone.FaceDetectionModule import FaceDetector
    except ImportError as exc:
        raise SystemExit(f'Missing face-collection dependency: {exc}. Install cvzone and mediapipe.') from exc
    args.output.mkdir(parents=True, exist_ok=True)
    detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

    def save_image(frame):
        if frame is None:
            return None
        labels, preview = collect_one(frame, detector, args.class_id,
                                      args.min_score, args.blur_threshold)
        if labels:
            name = str(time_ns())
            if not cv2.imwrite(str(args.output / (name + '.jpg')), frame):
                raise RuntimeError('Failed to save captured image')
            (args.output / (name + '.txt')).write_text('\n'.join(labels) + '\n', encoding='utf-8')
        return preview

    source = Path(args.source)
    if source.is_dir():
        n = 0
        for path in sorted(source.iterdir()):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                continue
            frame = cv2.imread(str(path))
            if frame is not None:
                save_image(frame)
                n += 1
        print(f'Processed {n} images. Private output folder: {args.output}')
        return
    if not args.source.isdecimal():
        raise SystemExit('Source must be a camera index or image folder')
    cap = cv2.VideoCapture(int(args.source))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open webcam {args.source}')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            preview = save_image(frame)
            if preview is not None:
                cv2.imshow('Collect faces — q to quit', preview)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
