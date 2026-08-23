from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(slots=True)
class CameraMotion:
    homography_current_to_previous: list[list[float]] | None
    homography_current_to_reference: list[list[float]] | None
    inlier_ratio: float
    pan_x: float
    pan_y: float
    zoom: float
    reliable: bool
    reset: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class CameraStabilizer:
    """Estimate pan/tilt/zoom motion and map points to a stable shot reference.

    This compensates camera movement inside one continuous shot. Metric pitch
    coordinates still require field-keypoint calibration through ``PitchProjector``.
    """

    def __init__(self, max_features: int = 900, min_matches: int = 18):
        import cv2

        self.detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = min_matches
        self.previous_gray = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.current_to_reference = np.eye(3, dtype=np.float64)

    def reset(self) -> None:
        self.previous_gray = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.current_to_reference = np.eye(3, dtype=np.float64)

    def update(self, frame, *, scene_cut: bool = False) -> CameraMotion:
        import cv2

        if scene_cut:
            self.reset()
        height, width = frame.shape[:2]
        scale = min(1.0, 960.0 / max(width, 1))
        working = frame
        if scale < 1:
            working = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if self.previous_descriptors is None or descriptors is None:
            self.previous_gray = gray
            self.previous_keypoints = keypoints
            self.previous_descriptors = descriptors
            return CameraMotion(
                homography_current_to_previous=None,
                homography_current_to_reference=self.current_to_reference.tolist(),
                inlier_ratio=0.0,
                pan_x=0.0,
                pan_y=0.0,
                zoom=1.0,
                reliable=False,
                reset=scene_cut,
            )

        matches = self.matcher.match(descriptors, self.previous_descriptors)
        matches = sorted(matches, key=lambda item: item.distance)[:180]
        reliable = len(matches) >= self.min_matches
        homography = None
        inlier_ratio = 0.0
        pan_x = pan_y = 0.0
        zoom = 1.0
        if reliable:
            current_points = np.float32([keypoints[item.queryIdx].pt for item in matches])
            previous_points = np.float32(
                [self.previous_keypoints[item.trainIdx].pt for item in matches]
            )
            homography, mask = cv2.findHomography(
                current_points,
                previous_points,
                cv2.RANSAC,
                3.0,
            )
            if homography is None or mask is None:
                reliable = False
            else:
                inlier_ratio = float(mask.ravel().mean())
                reliable = inlier_ratio >= 0.34
                if reliable:
                    self.current_to_reference = self.current_to_reference @ homography
                    pan_x = float(homography[0, 2] / max(scale, 1e-6))
                    pan_y = float(homography[1, 2] / max(scale, 1e-6))
                    zoom = float((abs(homography[0, 0]) + abs(homography[1, 1])) / 2.0)

        self.previous_gray = gray
        self.previous_keypoints = keypoints
        self.previous_descriptors = descriptors
        return CameraMotion(
            homography_current_to_previous=homography.tolist() if homography is not None else None,
            homography_current_to_reference=self.current_to_reference.tolist(),
            inlier_ratio=round(inlier_ratio, 4),
            pan_x=round(pan_x, 3),
            pan_y=round(pan_y, 3),
            zoom=round(zoom, 5),
            reliable=reliable,
            reset=scene_cut,
        )

    def stabilize_point(self, x: float, y: float) -> tuple[float, float]:
        point = np.array([x, y, 1.0], dtype=np.float64)
        projected = self.current_to_reference @ point
        if abs(projected[2]) < 1e-8:
            return x, y
        return float(projected[0] / projected[2]), float(projected[1] / projected[2])


class PitchProjector:
    PITCH_LENGTH_M = 105.0
    PITCH_WIDTH_M = 68.0

    def __init__(self):
        self.homography = None
        self.reprojection_error = None

    def calibrate(
        self,
        image_points: list[tuple[float, float]],
        pitch_points: list[tuple[float, float]],
    ) -> bool:
        import cv2

        if len(image_points) < 4 or len(image_points) != len(pitch_points):
            return False
        image = np.asarray(image_points, dtype=np.float32)
        pitch = np.asarray(pitch_points, dtype=np.float32)
        homography, mask = cv2.findHomography(image, pitch, cv2.RANSAC, 4.0)
        if homography is None:
            return False
        self.homography = homography
        projected = cv2.perspectiveTransform(image.reshape(-1, 1, 2), homography).reshape(-1, 2)
        self.reprojection_error = float(np.linalg.norm(projected - pitch, axis=1).mean())
        return True

    def project(self, x: float, y: float) -> tuple[float, float] | None:
        import cv2

        if self.homography is None:
            return None
        point = np.array([[[x, y]]], dtype=np.float32)
        result = cv2.perspectiveTransform(point, self.homography)[0, 0]
        return float(result[0]), float(result[1])
