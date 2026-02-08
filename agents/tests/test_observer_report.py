import numpy as np

from agents.utils import observer_report as report


def test_diff_score_identical_is_zero():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert report.diff_score(img, img) == 0.0


def test_diff_score_change_is_positive():
    img_a = np.zeros((64, 64, 3), dtype=np.uint8)
    img_b = np.full((64, 64, 3), 255, dtype=np.uint8)
    score = report.diff_score(img_a, img_b)
    assert score is not None
    assert score > 0.5


def test_diff_score_missing_returns_none():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert report.diff_score(None, img) is None
