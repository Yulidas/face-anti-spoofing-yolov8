#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-time face anti-spoofing inference with YOLOv8 RGB and HSV models.

Runs either detector independently or combines spatially matched detections.
In combined mode, REAL requires agreement from both models; FAKE takes
precedence on disagreement. Includes temporal confirmation and a compact
OpenCV webcam overlay. Detection scores are not calibrated probabilities.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Optional

import numpy as np

REAL = 'REAL'
FAKE = 'FAKE'
VERIFY = 'VERIFICATION'
PARTIAL = 'DETECTION PARTIELLE'
TOO_CLOSE = 'REJET - TROP PROCHE'
DEPTH_REJECT = 'REJET - RELIEF FAIBLE'
NO_FACE = 'AUCUN VISAGE'


@dataclass(frozen=True)
class Face:
    bbox: tuple[int, int, int, int]
    cls: int                   # 0=Fake, 1=Real
    conf: float                # confiance detection pour la classe preditE


@dataclass(frozen=True)
class Decision:
    bbox: tuple[int, int, int, int]
    status: str
    reason: str
    hsv: Optional[Face] = None
    rgb: Optional[Face] = None
    score: Optional[float] = None
    confirmations: int = 0
    depth_note: str = 'MiDaS: OFF'


@dataclass
class _Track:
    bbox: tuple[int, int, int, int]
    confirmations: int
    timestamp: float


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1-x0) * max(0, y1-y0)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    denom = aa + bb - overlap
    return overlap / denom if denom > 0 else 0.0


def area_fraction(bbox, shape) -> float:
    h, w = shape[:2]
    return max(0, bbox[2]-bbox[0]) * max(0, bbox[3]-bbox[1]) / max(1, h*w)


def match_faces(hsv: list[Face], rgb: list[Face], min_iou: float = 0.25):
    """Association geometrique stricte 1:1. Pas d'appariement par ordre."""
    pairs = sorted(((iou(h.bbox, r.bbox), i, j)
                    for i, h in enumerate(hsv) for j, r in enumerate(rgb)), reverse=True)
    used_h, used_r, matched = set(), set(), []
    for overlap, i, j in pairs:
        if overlap < min_iou:
            break
        if i not in used_h and j not in used_r:
            used_h.add(i)
            used_r.add(j)
            matched.append((hsv[i], rgb[j]))
    return (matched,
            [h for i, h in enumerate(hsv) if i not in used_h],
            [r for j, r in enumerate(rgb) if j not in used_r])


def check_model(model, label: str):
    names = model.names
    classes = [str(names.get(i, '') if isinstance(names, dict) else names[i]).strip().lower()
               for i in (0, 1)]
    if classes != ['fake', 'real']:
        raise ValueError(f'{label}: classes {names!r}; attendu 0=Fake, 1=Real')


def collapse_overlaps(faces: list[Face], overlap_min: float = .65) -> list[Face]:
    """Supprime les boites dupliquees, en conservant un avis FAKE credible.

    Les detections classes Fake et Real se chevauchent parfois (NMS par
    classe). Les afficher comme deux personnes serait trompeur.
    """
    output: list[Face] = []
    for face in sorted(faces, key=lambda f: f.conf, reverse=True):
        pos = next((i for i, x in enumerate(output) if iou(face.bbox, x.bbox) >= overlap_min), None)
        if pos is None:
            output.append(face)
        elif face.cls == 0 and output[pos].cls == 1 and face.conf >= .40:
            output[pos] = face
    return output


def extract_faces(result) -> list[Face]:
    faces: list[Face] = []
    if getattr(result, 'boxes', None) is None:
        return faces
    for b in result.boxes:
        c, conf = int(b.cls[0].item()), float(b.conf[0].item())
        coords = tuple(int(round(x)) for x in b.xyxy[0].tolist())
        if c in (0, 1) and 0 <= conf <= 1 and coords[2] > coords[0] and coords[3] > coords[1]:
            faces.append(Face(coords, c, conf))
    return collapse_overlaps(faces)


def predict_faces(model, bgr: np.ndarray, mode: str, conf: float, device: str = 'auto') -> list[Face]:
    import cv2
    if mode not in ('hsv', 'rgb', 'bgr'):
        raise ValueError(f'Mode couleur inconnu: {mode}')
    # Attention: le modele RGB historique du projet lisait les images via
    # OpenCV. Pour respecter ce checkpoint, on transmet son BGR natif.
    inp = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) if mode == 'hsv' else bgr
    kw = {'conf': conf, 'verbose': False}
    if device != 'auto':
        kw['device'] = 0 if device == 'cuda' else 'cpu'
    return extract_faces(model.predict(inp, **kw)[0])


def classify(hsv: Optional[Face], rgb: Optional[Face], shape, *,
             mode: str = 'both', real_score_min: float = .50,
             max_face_area: float = .55,
             depth_contrast: Optional[float] = None,
             flat_threshold: float = .02,
             depth_required: bool = False,
             depth_note: str = 'MiDaS: OFF') -> Decision:
    """Decision lisible: Fake prioritaire, fusion Real strictement 50/50.

    En mode BOTH, un REAL necessite DEUX predictions REAL. Une FAKE
    detectee l'emporte, meme si l'autre modele voit REAL. Une detection
    manquante ne permet jamais de fabriquer artificiellement un REAL.
    """
    if hsv is None and rgb is None:
        raise ValueError('classify exige au moins une detection')
    anchor = hsv or rgb
    assert anchor is not None
    score: Optional[float] = None
    if mode == 'both' and hsv is not None and rgb is not None and hsv.cls == rgb.cls:
        score = .5 * hsv.conf + .5 * rgb.conf
    elif mode in ('hsv', 'rgb'):
        score = anchor.conf

    def result(status: str, reason: str) -> Decision:
        return Decision(anchor.bbox, status, reason, hsv, rgb, score=score, depth_note=depth_note)

    involved = [f for f in (hsv, rgb) if f is not None]
    # Priorite FAKE a toute acceptation REAL, meme sur photo rapprochee.
    if any(f.cls == 0 for f in involved):
        origin = '+'.join(name for name, f in [('HSV', hsv), ('RGB', rgb)]
                          if f is not None and f.cls == 0)
        return result(FAKE, f'{origin} detecte FAKE')

    # Test geometrique simple : garde-fou, pas estimation reelle distance.
    if area_fraction(anchor.bbox, shape) > max_face_area:
        return result(TOO_CLOSE, 'Visage trop grand dans la camera')

    if mode == 'both' and (hsv is None or rgb is None):
        return result(PARTIAL, 'Un des deux modeles ne detecte pas le visage')
    if mode == 'both' and (score is None or score < real_score_min):
        return result(VERIFY, f'Score REAL faible (minimum {real_score_min:.2f})')

    # MiDaS n'est jamais un signal positif de liveness. Un signal plat
    # peut bloquer, mais une estimation indisponible ne fait pas de faux bonus.
    if depth_required and depth_contrast is None:
        return result(VERIFY, 'Mesure MiDaS recente indisponible')
    if depth_contrast is not None and depth_contrast < flat_threshold:
        return result(DEPTH_REJECT, 'Pseudo-relief MiDaS faible')

    return result(REAL, 'Accord HSV + RGB' if mode == 'both' else f'{mode.upper()} confirme')


def decide_frame(hsv: list[Face], rgb: Optional[list[Face]], shape, args,
                 *, mode: str = 'both',
                 depth_values: Optional[dict] = None,
                 depth_notes: Optional[dict] = None,
                 depth_required: bool = False, flat_threshold: float = .02) -> list[Decision]:
    values = depth_values or {}
    notes = depth_notes or {}

    def classify_pair(h, r):
        anchor = h or r
        return classify(h, r, shape, mode=mode, real_score_min=args.real_score_min,
                        max_face_area=args.max_face_area,
                        depth_contrast=values.get(anchor.bbox),
                        depth_note=notes.get(anchor.bbox, 'MiDaS: OFF'),
                        depth_required=depth_required, flat_threshold=flat_threshold)

    if mode == 'hsv':
        return [classify_pair(h, None) for h in hsv]
    if mode == 'rgb':
        return [classify_pair(None, r) for r in (rgb or [])]
    paired, unmatched_h, unmatched_r = match_faces(hsv, rgb or [], args.iou_match)
    return ([classify_pair(h, r) for h, r in paired] +
            [classify_pair(h, None) for h in unmatched_h] +
            [classify_pair(None, r) for r in unmatched_r])


class TemporalGate:
    """REAL confirme sur N observations independantes de la meme zone.

    FAKE/REJET s'appliquent immediatement. Toute perte de correspondance
    temporelle annule la confirmation, pour eviter une approbation stale.
    """
    def __init__(self, required: int = 2, max_gap: float = 2.5):
        self.required = max(1, required)
        self.max_gap = max_gap
        self.tracks: list[_Track] = []

    def update(self, decisions: list[Decision], now: float) -> list[Decision]:
        old = [t for t in self.tracks if 0 <= now - t.timestamp <= self.max_gap]
        proposals = sorted(((iou(d.bbox, t.bbox), i, j)
                            for i, d in enumerate(decisions) for j, t in enumerate(old)), reverse=True)
        assigned, seen_d, seen_t = {}, set(), set()
        for overlap, i, j in proposals:
            if overlap < .40:
                break
            if i not in seen_d and j not in seen_t:
                assigned[i] = old[j]
                seen_d.add(i)
                seen_t.add(j)
        new_tracks, out = [], []
        for idx, d in enumerate(decisions):
            prev = assigned.get(idx)
            n = (prev.confirmations + 1 if prev else 1) if d.status == REAL else 0
            new_tracks.append(_Track(d.bbox, n, now))
            if d.status == REAL and n < self.required:
                out.append(replace(d, status=VERIFY, confirmations=n,
                                   reason=f'Validation {n}/{self.required}'))
            else:
                out.append(replace(d, confirmations=n))
        self.tracks = new_tracks
        return out


def face_label(face: Optional[Face], title: str) -> str:
    return (f'{title}  {REAL if face.cls == 1 else FAKE} {face.conf:.2f}'
            if face else f'{title}  --')


def status_color(status: str) -> tuple[int, int, int]:
    if status == REAL:
        return (42, 150, 42)
    if status == FAKE:
        return (52, 65, 210)
    if status in (TOO_CLOSE, DEPTH_REJECT):
        return (20, 115, 220)
    return (90, 100, 110)


def _rounded_rect(img, p1, p2, radius, fill):
    import cv2
    x0, y0 = p1; x1, y1 = p2
    r = max(0, min(radius, (x1-x0)//2, (y1-y0)//2))
    cv2.rectangle(img, (x0+r, y0), (x1-r, y1), fill, -1)
    cv2.rectangle(img, (x0, y0+r), (x1, y1-r), fill, -1)
    for x, y in [(x0+r, y0+r), (x1-r, y0+r), (x0+r, y1-r), (x1-r, y1-r)]:
        cv2.circle(img, (x,y), r, fill, -1, cv2.LINE_AA)


def annotate(frame: np.ndarray, decisions: list[Decision], interval: float,
             *, mode: str = 'both', depth_enabled: bool = False,
             time_since: float = 0.0) -> np.ndarray:
    """Interface compacte: carte blanche opaque 260x112, pas de bande plein ecran."""
    import cv2
    out = frame.copy()
    H, W = out.shape[:2]
    # Chaque visage recoit son propre verdict a proximite de la boite.
    for d in decisions:
        box = tuple(int(x) for x in d.bbox)
        color = status_color(d.status)
        cv2.rectangle(out, box[:2], box[2:], color, 2)
        x = max(4, box[0]); y = max(18, box[1] - 6)
        label = d.status if d.status in (REAL, FAKE) else 'REJET' if d.status in (TOO_CLOSE, DEPTH_REJECT) else 'CHECK'
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)

    main = max(decisions, key=lambda d: area_fraction(d.bbox, out.shape)) if decisions else None
    status = main.status if main else NO_FACE
    active = status_color(status)
    # Reduire la taille dans les petites resolutions; ne jamais couvrir tout l'ecran.
    panel_w = min(300, max(180, W - 20))
    line_h = 18 if depth_enabled else 19
    panel_h = 113 if depth_enabled else 95
    panel_h = min(panel_h, max(65, H - 20))
    x0 = max(8, W - panel_w - 12)
    y0 = 12 if H > panel_h + 15 else 5
    x1 = min(W - 5, x0 + panel_w)
    y1 = min(H - 4, y0 + panel_h)
    # Carte opaque claire (aucune bande sombre en pleine largeur).
    _rounded_rect(out, (x0, y0), (x1, y1), 11, (247, 249, 251))
    cv2.rectangle(out, (x0+1, y0+9), (x0+4, y1-9), active, -1)

    def draw(text, y, *, color=(55, 61, 69), bold=False, scale=.46):
        max_w = panel_w - 24
        thickness = 2 if bold else 1
        while scale > .30 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > max_w:
            scale -= .025
        cv2.putText(out, text, (x0+13, y), cv2.FONT_HERSHEY_SIMPLEX,
                    max(.30, scale), color, thickness, cv2.LINE_AA)

    text = status if main else NO_FACE
    if text == PARTIAL:
        text = 'DETECTION PARTIELLE'
    draw(text, y0+21, color=active, bold=True, scale=.53)
    if main:
        if mode in ('both', 'hsv'):
            hsv_text = face_label(main.hsv, 'HSV')
        else:
            hsv_text = 'HSV  OFF'
        if mode in ('both', 'rgb'):
            rgb_text = face_label(main.rgb, 'RGB')
        else:
            rgb_text = 'RGB  OFF'
        draw(hsv_text, y0+41)
        draw(rgb_text, y0+60)
        detail = (f'Fusion 50/50  {main.score:.2f}'
                  if mode == 'both' and main.status == REAL and main.score is not None
                  else main.reason)
        if depth_enabled:
            draw(main.depth_note, y0+78, scale=.41)
            draw(detail, y0+96, scale=.39)
        else:
            draw(detail, y0+78, scale=.41)
    else:
        draw('HSV  --', y0+41)
        draw('RGB  --', y0+60)
        if depth_enabled:
            draw('MiDaS  --', y0+78)
    return out


def make_parser():
    p = argparse.ArgumentParser(description='Face Anti-Spoofing V4 | Webcam HSV, RGB ou BOTH')
    p.add_argument('--mode', choices=('hsv', 'rgb', 'both'), default=None,
                   help='hsv, rgb ou both (fusion 50/50, FAKE prioritaire)')
    p.add_argument('--hsv-model', type=Path, help='Checkpoint HSV')
    p.add_argument('--rgb-model', type=Path, help='Checkpoint RGB historique (BGR OpenCV)')
    # Compatibilite avec les anciennes commandes du depot.
    p.add_argument('--model', type=Path, help='Ancien alias pour le checkpoint principal')
    p.add_argument('--secondary-model', type=Path, help='Ancien alias modele RGB')
    p.add_argument('--color', choices=('hsv', 'rgb'), default='hsv', help='Couleur de --model')
    p.add_argument('--input', default='0', help='Webcam 0, chemin image/video')
    p.add_argument('--conf', type=float, default=.25, help='Confiance minimale des boites YOLO')
    p.add_argument('--iou-match', type=float, default=.25)
    p.add_argument('--real-score-min', type=float, default=.50,
                   help='Score moyen minimum lorsque tous les modeles predisent REAL')
    p.add_argument('--max-face-area', type=float, default=.55,
                   help='Part de la surface image au-dela de laquelle le visage est trop proche')
    p.add_argument('--interval', type=float, default=.35, help='Pause minimale entre analyses')
    p.add_argument('--stable-real', type=int, default=2, help='Confirmations avant REAL')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--output', type=Path, help='Fichier image pour sortie sur une image')
    return p


def resolve_legacy_models(args):
    if args.model is not None:
        if args.color == 'hsv':
            args.hsv_model = args.hsv_model or args.model
        else:
            args.rgb_model = args.rgb_model or args.model
    if args.secondary_model is not None:
        args.rgb_model = args.rgb_model or args.secondary_model
    if args.mode is None:
        args.mode = ('both' if args.hsv_model and args.rgb_model else
                     'hsv' if args.hsv_model else 'rgb' if args.rgb_model else 'both')


def validate_args(args):
    resolve_legacy_models(args)
    if args.mode in ('hsv', 'both') and (args.hsv_model is None or not args.hsv_model.is_file()):
        raise SystemExit('Checkpoint HSV introuvable : --hsv-model models/....pt')
    if args.mode in ('rgb', 'both') and (args.rgb_model is None or not args.rgb_model.is_file()):
        raise SystemExit('Checkpoint RGB introuvable : --rgb-model models/....pt')
    for prop in ('conf', 'iou_match', 'real_score_min', 'max_face_area'):
        if not 0 <= getattr(args, prop) <= 1:
            raise SystemExit(f'--{prop.replace("_", "-")} doit etre entre 0 et 1')
    if args.interval < 0 or args.stable_real < 1:
        raise SystemExit('--interval >= 0 et --stable-real >= 1 requis')


def load_yolos(args):
    from ultralytics import YOLO
    hsv = YOLO(str(args.hsv_model)) if args.mode in ('hsv', 'both') else None
    rgb = YOLO(str(args.rgb_model)) if args.mode in ('rgb', 'both') else None
    if hsv is not None:
        check_model(hsv, 'HSV')
    if rgb is not None:
        check_model(rgb, 'RGB')
    return hsv, rgb


def process_frame(frame, hsv_net, rgb_net, args):
    hsv = predict_faces(hsv_net, frame, 'hsv', args.conf, args.device) if hsv_net else []
    rgb = predict_faces(rgb_net, frame, 'bgr', args.conf, args.device) if rgb_net else []
    return decide_frame(hsv, rgb, frame.shape, args, mode=args.mode)


def run_camera(args, hsv_net, rgb_net):
    import cv2
    src = int(args.input) if args.input.isdecimal() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f'Impossible d ouvrir webcam/video: {args.input}')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    gate = TemporalGate(args.stable_real, max_gap=max(2.0, args.interval*4))
    last_at = -float('inf')
    decisions: list[Decision] = []
    print(f'Webcam {args.mode.upper()} ouverte. Q/Echap pour quitter.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if time.monotonic() - last_at >= args.interval:
                decisions = gate.update(process_frame(frame, hsv_net, rgb_net, args), time.monotonic())
                last_at = time.monotonic()
            age = max(0, time.monotonic() - last_at)
            visible = decisions if age < max(1.0, args.interval * 3) else []
            cv2.imshow(f'Anti-Spoofing | {args.mode.upper()} | Q pour quitter',
                       annotate(frame, visible, args.interval, mode=args.mode, time_since=age))
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = make_parser().parse_args()
    validate_args(args)
    try:
        import cv2
        from ultralytics import YOLO  # noqa:F401
    except ImportError as exc:
        raise SystemExit(f'Dependance manquante: {exc}') from exc
    hsv, rgb = load_yolos(args)
    item = Path(args.input)
    if item.is_file() and item.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}:
        frame = cv2.imread(str(item))
        if frame is None:
            raise SystemExit(f'Image illisible: {item}')
        result = annotate(frame, process_frame(frame, hsv, rgb, args), args.interval, mode=args.mode)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.output), result):
                raise SystemExit('Enregistrement image echoue')
        else:
            cv2.imshow('Anti-Spoofing', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        if args.output:
            raise SystemExit('--output accepte seulement pour une image dans detect.py')
        if not args.input.isdecimal() and not item.is_file():
            raise SystemExit(f'Source introuvable: {args.input}')
        run_camera(args, hsv, rgb)


if __name__ == '__main__':
    main()
