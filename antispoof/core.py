"""Lightweight post-processing utilities for face anti-spoofing.

Defines validated detection records, bounding-box overlap calculations,
spatial matching, and simple decision-fusion helpers. These operations
are independent of model-loading and camera libraries.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[int, int, int, int]
    label: int  # training convention: 0=Fake, 1=Real
    score: float

    def __post_init__(self):
        if self.label not in (0, 1):
            raise ValueError('Expected labels: 0=Fake, 1=Real. Check your model metadata.')
        if not (0.0 <= self.score <= 1.0):
            raise ValueError('Confidence score must be within [0, 1]')
        x1, y1, x2, y2 = self.xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Bounding box must have positive area')


def intersection_over_union(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return intersection / (a_area + b_area - intersection)


def associate_detections(primary: list[Detection], secondary: list[Detection], min_iou: float = 0.25):
    """Greedy spatial association; unlike zip(), doesn't rely on ordering of boxes.

    Returns matched (primary, secondary) pairs and unmatched primary detections.
    This is a demo heuristic rather than a multi-object tracker.
    """
    if not 0 <= min_iou <= 1:
        raise ValueError('min_iou must be between 0 and 1')
    candidates = sorted(
        ((intersection_over_union(p, s), i, j) for i, p in enumerate(primary)
         for j, s in enumerate(secondary)), reverse=True
    )
    used_primary, used_secondary, matched = set(), set(), []
    for iou, i, j in candidates:
        if iou < min_iou:
            break
        if i not in used_primary and j not in used_secondary:
            matched.append((primary[i], secondary[j]))
            used_primary.add(i)
            used_secondary.add(j)
    unmatched = [p for i, p in enumerate(primary) if i not in used_primary]
    return matched, unmatched


def fuse_predictions(primary: Detection, secondary: Detection) -> tuple[str, float]:
    """Historical conservative disagreement rule: Fake if models disagree.

    Returns display label and heuristic score. Scores are *not* calibrated probabilities.
    """
    if primary.label != secondary.label:
        return 'Fake (disagreement)', max(primary.score, secondary.score)
    return ('Real' if primary.label == 1 else 'Fake'), (primary.score + secondary.score) / 2


def display_label(d: Detection) -> str:
    return 'Real' if d.label == 1 else 'Fake'
