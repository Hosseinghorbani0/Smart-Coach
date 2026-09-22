from types import SimpleNamespace

import pytest

pytest.importorskip("mediapipe")

from processes.exercise_detector import ExerciseDetector


def make_landmarks():
    landmarks = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(33)]

    # Both sides are straight, so knee and elbow angles should be 180 degrees.
    for shoulder, elbow, wrist in ((11, 13, 15), (12, 14, 16)):
        landmarks[shoulder].y = 0.0
        landmarks[elbow].y = 0.5
        landmarks[wrist].y = 1.0

    for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
        landmarks[hip].y = 0.0
        landmarks[knee].y = 0.5
        landmarks[ankle].y = 1.0

    landmarks[7].y = -0.2
    return landmarks


def make_detector():
    detector = ExerciseDetector.__new__(ExerciseDetector)
    detector.last_angles = {}
    detector.angle_smoothing = 0.35
    return detector


def test_key_angles_average_both_body_sides():
    detector = make_detector()

    angles = detector.detect_key_angles(make_landmarks())

    assert angles["knee"] == pytest.approx(180.0)
    assert angles["elbow"] == pytest.approx(180.0)
    assert angles["chin_to_shoulder"] == pytest.approx(-0.2)


def test_low_visibility_side_is_ignored():
    detector = make_detector()
    landmarks = make_landmarks()
    landmarks[24].visibility = 0.1
    landmarks[26].visibility = 0.1
    landmarks[28].visibility = 0.1

    angles = detector.detect_key_angles(landmarks)

    assert angles["knee"] == pytest.approx(180.0)
