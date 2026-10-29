import numpy as np

from tomato_model_utils import (
    estimate_affected_area,
    estimate_severity,
    gradcam_explanation,
)


def test_estimate_affected_area_for_green_leaf_is_low():
    green_leaf = np.zeros((32, 32, 3), dtype=np.uint8)
    green_leaf[:, :, 1] = 200
    green_leaf[:, :, 0] = 50
    green_leaf[:, :, 2] = 50

    assert estimate_affected_area(green_leaf) < 10


def test_estimate_severity_returns_severe_for_confident_disease():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    severity = estimate_severity('Late-blight', 0.96, image)
    assert severity == 'Severe'


def test_gradcam_explanation_mentions_heatmap_and_symptoms():
    explanation = gradcam_explanation('Late-blight', 'Severe', 38.4)
    assert 'Grad-CAM' in explanation
    assert 'leaf regions' in explanation.lower()
    assert 'severity' in explanation.lower()
