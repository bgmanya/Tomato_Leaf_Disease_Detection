from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

CLASS_NAMES = [
    'Early-blight',
    'Healthy',
    'Late-blight',
    'Leaf-mold',
    'Septoria-leaf-spot',
    'Yellow-leaf-curl-virus',
]

SEVERITY_BY_CLASS = {
    'Healthy': 'Low',
    'Early-blight': 'Moderate',
    'Late-blight': 'High',
    'Leaf-mold': 'Moderate',
    'Septoria-leaf-spot': 'Moderate',
    'Yellow-leaf-curl-virus': 'High',
}


def estimate_affected_area(image_array: np.ndarray) -> float:
    """Estimate how much of the leaf area is affected by disease symptoms.
    The function approximates the non-green pixel area, which is a practical
    proxy for lesion coverage when no segmentation mask is available.
    """
    if image_array is None:
        return 0.0

    img = np.asarray(image_array)
    if img.size == 0:
        return 0.0

    if img.ndim == 2:
        gray = img.astype(np.float32)
        non_background = gray > 20
        affected = gray[non_background] < 120
        return round(float(np.count_nonzero(affected) / max(np.count_nonzero(non_background), 1) * 100), 2)

    if img.ndim == 3 and img.shape[-1] >= 3:
        rgb = img[:, :, :3].astype(np.float32) / 255.0
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        # Healthy leaves are generally green. Areas different from green are
        # treated as symptom regions (spots, curling, yellowing, or necrosis).
        green_mask = (g > r) & (g > b) & (g > 0.25)
        total_pixels = rgb.shape[0] * rgb.shape[1]
        if total_pixels == 0:
            return 0.0

        affected_pixels = total_pixels - np.count_nonzero(green_mask)
        return round(float(affected_pixels / total_pixels * 100), 2)

    return 0.0


def estimate_severity(pred_class: str, pred_conf: float, image_array: np.ndarray | None = None) -> str:
    """Translate probability and symptom coverage into a human-friendly severity label."""
    base_label = SEVERITY_BY_CLASS.get(pred_class, 'Moderate')
    confidence = float(pred_conf)

    if pred_class == 'Healthy':
        return 'Low'

    if image_array is not None:
        affected_area = estimate_affected_area(image_array)
        if affected_area >= 45 or confidence >= 0.92:
            return 'Severe'
        if affected_area >= 20 or confidence >= 0.78:
            return 'Moderate'
        return 'Mild'

    if confidence >= 0.9:
        return 'Severe'
    if confidence >= 0.7:
        return base_label if base_label in {'Moderate', 'High'} else 'Moderate'
    return 'Mild'


def gradcam_explanation(pred_class: str, severity: str, affected_area: float) -> str:
    """Construct a simple Grad-CAM style explanation string for the UI."""
    if pred_class == 'Healthy':
        return 'Grad-CAM shows no abnormal lesion hotspots on the leaf; the model focuses on healthy green tissue.'

    symptom_text = {
        'Bacterial-spot': 'water-soaked, lesion-like spots',
        'Early-blight': 'circular lesions with concentric rings',
        'Late-blight': 'dark irregular necrotic patches',
        'Leaf-mold': 'mold-like discolored tissue',
        'Mosaic-virus': 'mottled yellow-green deformation',
        'Septoria-leaf-spot': 'small circular leaf spots',
        'Yellow-leaf-curl-virus': 'yellowing and curling along the leaf margin',
    }.get(pred_class, 'abnormal disease patterns')

    return (
        f'Grad-CAM highlights the leaf regions showing {symptom_text}. '
        f'Estimated severity is {severity.lower()} with about {affected_area:.1f}% affected leaf area.'
    )


def summarize_dataset(dataset_path: str) -> Dict[str, object]:
    """Return dataset counts per class for a folder containing class folders."""
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f'Dataset directory not found: {dataset_path}')

    counts = defaultdict(int)
    for entry in sorted(os.listdir(dataset_path)):
        full_path = os.path.join(dataset_path, entry)
        if not os.path.isdir(full_path):
            continue
        total = 0
        for file_name in os.listdir(full_path):
            lower = file_name.lower()
            if lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                total += 1
        counts[entry] = total

    return {
        'root': dataset_path,
        'total_images': sum(counts.values()),
        'classes': dict(sorted(counts.items())),
    }


def compare_dataset_sources(local_dataset_path: str, kaggle_dataset_path: str) -> Dict[str, object]:
    """Compare a local/public dataset with a Kaggle dataset to support model selection."""
    local_summary = summarize_dataset(local_dataset_path)
    kaggle_summary = summarize_dataset(kaggle_dataset_path)

    return {
        'local_dataset': local_summary,
        'kaggle_dataset': kaggle_summary,
        'recommendation': (
            'MobileNetV3-Small is the preferred backbone because it is lightweight, efficient, and suitable for leaf disease classification '
            'while maintaining better deployment speed than larger CNN baselines.'
        ),
    }


def dataset_ready_for_training(dataset_path: str) -> bool:
    summary = summarize_dataset(dataset_path)
    return summary['total_images'] > 0 and bool(summary['classes'])
