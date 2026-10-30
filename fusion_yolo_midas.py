#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimental face anti-spoofing fusion using YOLOv8 and MiDaS.

Combines HSV and standard-color YOLO detections with an optional monocular
pseudo-depth cue. Reuses the conservative decision logic from detect.py
and supports selective depth estimation for live webcam inference.
MiDaS depth estimates are heuristic, not physical proof of liveness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import numpy as np

from detect import (Face, TemporalGate, annotate, check_model, decide_frame,
                    iou, match_faces, predict_faces)


class DepthEstimator:
    """MiDaS via torch.hub : code tiers charge depuis ./MiDaS ou upstream."""
    def __init__(self, model_type: str, device: str, local_repo: Optional[Path]):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if local_repo is not None:
            if not (local_repo / 'hubconf.py').is_file():
                raise FileNotFoundError(f'MiDaS/hubconf.py introuvable: {local_repo}')
            repo = str(local_repo.resolve())
            kwargs = {'source': 'local'}
        else:
            repo = 'intel-isl/MiDaS'
            kwargs = {'trust_repo': True}
        self.model = torch.hub.load(repo, model_type, **kwargs).to(self.device).eval()
        transforms = torch.hub.load(repo, 'transforms', **kwargs)
        self.transform = (transforms.small_transform if model_type == 'MiDaS_small'
                          else transforms.dpt_transform)

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        import cv2
        torch = self.torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        batch = self.transform(rgb).to(self.device)
        with torch.inference_mode():
            pred = self.model(batch)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            if pred.ndim == 4:
                pred = pred.squeeze(1)
            resized = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=rgb.shape[:2], mode='bicubic',
                align_corners=False).squeeze(1).squeeze(0)
        return resized.float().cpu().numpy()


def depth_roi(depth: np.ndarray, bbox, xa: float, xb: float, ya: float, yb: float):
    h, w = depth.shape
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(w-1, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h-1, y0)); y1 = max(0, min(h, y1))
    if x1-x0 < 25 or y1-y0 < 25:
        return np.empty(0, dtype=np.float32)
    lx = max(0, min(w-1, x0 + int((x1-x0)*xa)))
    ux = max(lx+1, min(w, x0 + int((x1-x0)*xb)))
    ly = max(0, min(h-1, y0 + int((y1-y0)*ya)))
    uy = max(ly+1, min(h, y0 + int((y1-y0)*yb)))
    return depth[ly:uy, lx:ux]


def relative_curvature(depth: np.ndarray, bbox) -> Optional[float]:
    """Contraste nez-joues normalise: heuristique, pas distance metrique."""
    face = depth_roi(depth, bbox, .13, .87, .22, .82)
    nose = depth_roi(depth, bbox, .42, .58, .43, .65)
    cheek_a = depth_roi(depth, bbox, .20, .34, .45, .68)
    cheek_b = depth_roi(depth, bbox, .66, .80, .45, .68)
    if any(x.size < 4 or not np.isfinite(x).all() for x in (face, nose, cheek_a, cheek_b)):
        return None
    low, high = np.percentile(face, (10, 90))
    span = float(high - low)
    if span < 1e-5:
        # Sortie valide mais quasi constante: indice de surface plate.
        return 0.0
    cheeks = (float(np.median(cheek_a)) + float(np.median(cheek_b))) / 2
    return float(np.clip((float(np.median(nose)) - cheeks) / span, -2, 2))


def view_depth(depth: np.ndarray) -> np.ndarray:
    import cv2
    valid = depth[np.isfinite(depth)]
    if not valid.size:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    low, high = np.percentile(valid, (2, 98))
    norm = np.nan_to_num(np.clip((depth - low) / max(float(high - low), 1e-5), 0, 1))
    return cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_TURBO)


@dataclass(frozen=True)
class DepthReading:
    bbox: tuple[int, int, int, int]
    contrast: Optional[float]
    timestamp: float


class DepthCache:
    def __init__(self):
        self.entries: list[DepthReading] = []
        self.last_full_eval = -float("inf")

    def put(self, faces: list[Face], depth: np.ndarray, now: float):
        self.last_full_eval = now
        self.entries = [DepthReading(face.bbox, relative_curvature(depth, face.bbox), now)
                        for face in faces]

    def get(self, bbox, now: float, ttl: float) -> Optional[DepthReading]:
        valid = [(iou(bbox, e.bbox), e) for e in self.entries
                 if now - e.timestamp <= ttl]
        best = max(valid, key=lambda x: x[0], default=(0.0, None))
        return best[1] if best[0] >= 0.55 else None


def make_parser():
    p = argparse.ArgumentParser(description='Fusion anti-spoofing V4 YOLO 50/50 + MiDaS')
    p.add_argument('--hsv-model', type=Path, required=True)
    p.add_argument('--rgb-model', type=Path, required=True)
    p.add_argument('--source', default='0', help='Index webcam, image ou video')
    p.add_argument('--midas-repo', type=Path, help='Depot local MiDaS: ./MiDaS')
    p.add_argument('--midas-model', choices=('MiDaS_small', 'DPT_Hybrid', 'DPT_Large'),
                   default='MiDaS_small')
    p.add_argument('--depth-mode', choices=('selective', 'always', 'off'), default='selective')
    p.add_argument('--depth-every', type=int, default=4,
                   help='Tous les N passages YOLO en mode selective')
    p.add_argument('--depth-min-interval', type=float, default=1.2)
    p.add_argument('--depth-cache-max-age', type=float, default=1.4)
    p.add_argument('--flat-threshold', type=float, default=.015,
                   help='Seuil experimental de relief estime (pas metrique)')
    p.add_argument('--require-depth-for-real', action='store_true',
                   help='Exige une mesure MiDaS recente pour afficher REAL')
    p.add_argument('--show-depth', action='store_true')
    p.add_argument('--depth-weight', type=float, default=None,
                   help='Compatibilite: 0 desactive MiDaS; aucun bonus REAL')
    p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--iou-match', type=float, default=.25)
    p.add_argument('--real-score-min', type=float, default=.50)
    p.add_argument('--max-face-area', type=float, default=.55)
    p.add_argument('--interval', type=float, default=.35)
    p.add_argument('--stable-real', type=int, default=2)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--output', type=Path, help='Image ou video .mp4 annotee')
    return p


def validate_args(args):
    for item in ('hsv_model', 'rgb_model'):
        file = getattr(args, item)
        if not file.is_file():
            raise SystemExit(f'{item}: checkpoint introuvable : {file}')
    for name in ('conf','iou_match','real_score_min','max_face_area'):
        if not 0 <= getattr(args, name) <= 1:
            raise SystemExit(f'--{name.replace("_", "-")} doit etre dans [0,1]')
    if args.interval < 0 or args.stable_real < 1:
        raise SystemExit('Intervalle >= 0 et stable-real >= 1 requis')
    if args.depth_every < 1 or args.depth_min_interval < 0 or args.depth_cache_max_age <= 0:
        raise SystemExit('Parametres de profondeur invalides')
    if args.depth_weight is not None:
        if not 0 <= args.depth_weight <= 1:
            raise SystemExit('--depth-weight: attendu entre 0 et 1')
        if args.depth_weight == 0:
            args.depth_mode = 'off'
    if args.depth_mode == 'off' and (args.show_depth or args.require_depth_for_real):
        raise SystemExit('--show-depth/--require-depth-for-real incompatibles avec --depth-mode off')


def process_rgb_hsv(frame, model_h, model_r, args, depth_net,
                    cache: DepthCache, index: int, now: float):
    faces_h = predict_faces(model_h, frame, 'hsv', args.conf, args.device)
    faces_r = predict_faces(model_r, frame, 'bgr', args.conf, args.device)
    matched, lost_h, lost_r = match_faces(faces_h, faces_r, args.iou_match)

    # MiDaS sert surtout aux cas difficiles + controle periodique.
    difficult = (bool(lost_h or lost_r) or any(
        h.cls != r.cls or h.conf < .65 or r.conf < .65 for h, r in matched))
    due = index % args.depth_every == 0
    can_run = (depth_net is not None and bool(faces_h) and
               now - cache.last_full_eval >= args.depth_min_interval)
    should_run = can_run and (args.depth_mode == 'always' or due or difficult)
    depth_map = None
    if should_run:
        depth_map = depth_net.predict(frame)
        # Marquer l'heure de fin pour ne pas reutiliser une mesure trop vieille.
        cache.put(faces_h, depth_map, time.monotonic())

    depths, notes = {}, {}
    current = time.monotonic()
    for h in faces_h:
        reading = cache.get(h.bbox, current, args.depth_cache_max_age)
        if args.depth_mode == 'off':
            notes[h.bbox] = 'MiDaS  OFF'
        elif reading is None or reading.contrast is None:
            notes[h.bbox] = 'MiDaS  indisponible'
        else:
            depths[h.bbox] = reading.contrast
            tag = 'faible' if reading.contrast < args.flat_threshold else 'mesure'
            notes[h.bbox] = f'MiDaS  {tag} {reading.contrast:+.3f}'

    decisions = decide_frame(faces_h, faces_r, frame.shape, args, mode='both',
                             depth_values=depths, depth_notes=notes,
                             depth_required=args.require_depth_for_real,
                             flat_threshold=args.flat_threshold)
    return decisions, depth_map


def main():
    args = make_parser().parse_args()
    validate_args(args)
    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(f'Dependance manquante: {e}') from e
    device = ('cpu' if args.device == 'cpu' else
              'cuda' if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and device != 'cuda':
        raise SystemExit('CUDA indisponible, utiliser --device cpu')
    hsv = YOLO(str(args.hsv_model))
    rgb = YOLO(str(args.rgb_model))
    check_model(hsv, 'HSV')
    check_model(rgb, 'RGB')
    midas = None
    if args.depth_mode != 'off':
        local_repo = args.midas_repo
        if local_repo is None and Path('MiDaS/hubconf.py').is_file():
            local_repo = Path('MiDaS')
        print(f'Chargement MiDaS ({args.midas_model}), source: {local_repo or "torch.hub"}')
        try:
            midas = DepthEstimator(args.midas_model, device, local_repo)
        except Exception as e:
            raise SystemExit(f'Chargement MiDaS echoue: {e}') from e

    cache = DepthCache()
    gate = TemporalGate(args.stable_real, max_gap=max(2.5, args.interval*4))
    src_file = Path(args.source)
    image_mode = src_file.is_file() and src_file.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}
    if image_mode:
        frame = cv2.imread(str(src_file))
        if frame is None:
            raise SystemExit(f'Image illisible: {src_file}')
        result, depth_map = process_rgb_hsv(frame, hsv, rgb, args, midas, cache, 0, time.monotonic())
        out = annotate(frame, result, args.interval, mode='both', depth_enabled=args.depth_mode != 'off')
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.output), out):
                raise SystemExit('Enregistrement image echoue')
        else:
            cv2.imshow('Face Anti-Spoofing', out)
            if args.show_depth and depth_map is not None:
                cv2.imshow('MiDaS pseudo-profondeur', view_depth(depth_map))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    if not args.source.isdecimal() and not src_file.is_file():
        raise SystemExit(f'Source introuvable: {args.source}')
    src = int(args.source) if args.source.isdecimal() else str(src_file)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f'Impossible d ouvrir la camera/video: {args.source}')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    writer = None
    last_at = -float('inf')
    count = 0
    shown_decisions = []
    last_depth_view = None
    print('Webcam Fusion V4 ouverte. Q ou Echap pour quitter.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if time.monotonic() - last_at >= args.interval:
                raw, depth_map = process_rgb_hsv(frame, hsv, rgb, args, midas,
                                                   cache, count, time.monotonic())
                shown_decisions = gate.update(raw, time.monotonic())
                if depth_map is not None and args.show_depth:
                    last_depth_view = view_depth(depth_map)
                count += 1
                last_at = time.monotonic()
            age = max(0., time.monotonic()-last_at)
            visible = shown_decisions if age < max(1., args.interval*3) else []
            out = annotate(frame, visible, args.interval, mode='both',
                           depth_enabled=args.depth_mode != 'off', time_since=age)
            if args.output and writer is None:
                if args.output.suffix.lower() != '.mp4':
                    raise SystemExit('En webcam, --output doit se terminer par .mp4')
                args.output.parent.mkdir(parents=True, exist_ok=True)
                fps = cap.get(cv2.CAP_PROP_FPS)
                fps = fps if 1 <= fps <= 120 else 20.
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (out.shape[1],out.shape[0]))
                if not writer.isOpened():
                    raise SystemExit('Impossible de creer la sortie mp4')
            if writer is not None:
                writer.write(out)
            cv2.imshow('Anti-Spoofing YOLO + MiDaS | Q pour quitter', out)
            if args.show_depth and last_depth_view is not None:
                cv2.imshow('MiDaS: pseudo-profondeur estimee', last_depth_view)
            if cv2.waitKey(1) & 0xff in (ord('q'),27):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
