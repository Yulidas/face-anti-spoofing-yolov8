#!/usr/bin/env python3
"""Prepare a YOLOv8 dataset for face anti-spoofing experiments.

Matches images with annotation files, creates reproducible train,
validation, and test subsets, and generates the data.yaml configuration.
Splits are random and are not guaranteed to be identity-disjoint.
"""
import argparse
from pathlib import Path
import random
import shutil

SPLITS = ('train', 'val', 'test')
EXTENSIONS = ('.jpg', '.jpeg', '.png')


def discover_pairs(root: Path):
    """Return (image, label) pairs, one per image stem; reject ambiguous names."""
    pairs = []
    stems = set()
    for image in sorted(root.iterdir()):
        if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
            continue
        if image.stem in stems:
            raise ValueError(f'Duplicate image stem {image.stem} (different extensions?)')
        stems.add(image.stem)
        label = image.with_suffix('.txt')
        if not label.exists():
            raise ValueError(f'Missing label: {label}')
        pairs.append((image, label))
    if not pairs:
        raise ValueError(f'No labeled image pairs in {root}')
    return pairs


def split_counts(n: int, train: float, val: float):
    if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
        raise ValueError('Expected 0 < train, val, test < 1')
    a, b = int(n * train), int(n * val)
    return a, b, n - a - b


def make_split(source: Path, destination: Path, seed=42, train=0.70, val=0.15, overwrite=False):
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f'Dataset input missing: {source}')
    if source == destination or source in destination.parents and source.parent == destination:
        raise ValueError('Invalid destination')
    if source in destination.parents or destination in source.parents:
        raise ValueError('Input/output folders must be disjoint')
    pairs = discover_pairs(source)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    ntrain, nval, ntest = split_counts(len(pairs), train, val)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f'{destination} exists; use --overwrite to replace it intentionally')
        if destination == Path('/') or len(destination.parts) <= 2:
            raise ValueError('Unsafe destination for deletion')
        shutil.rmtree(destination)
    groups = dict(zip(SPLITS, (pairs[:ntrain], pairs[ntrain:ntrain+nval], pairs[ntrain+nval:])))
    for name, subset in groups.items():
        images, labels = destination / name / 'images', destination / name / 'labels'
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for image, label in subset:
            shutil.copy2(image, images / image.name)
            shutil.copy2(label, labels / label.name)
    config = (
        f'path: {destination.as_posix()}\n'
        'train: train/images\nval: val/images\ntest: test/images\n'
        'names:\n  0: Fake\n  1: Real\n'
    )
    (destination / 'data.yaml').write_text(config, encoding='utf-8')
    return {key: len(value) for key, value in groups.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('dataset/All'))
    p.add_argument('--output', type=Path, default=Path('dataset/SplitData'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train', type=float, default=0.70)
    p.add_argument('--val', type=float, default=0.15)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    counts = make_split(args.input, args.output, args.seed, args.train, args.val, args.overwrite)
    print('Prepared split (image/label pairs):', counts)
    print('IMPORTANT: separate individuals/sessions across splits to prevent data leakage.')


if __name__ == '__main__':
    main()
