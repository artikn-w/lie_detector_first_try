"""Webcam measurements and optional labeled trial recording."""
import argparse
import csv
import time
from collections import deque
from pathlib import Path
from statistics import median

import cv2
import mediapipe as mp
import numpy as np


FEATURES = ("blinks_per_minute", "head_speed_deg_per_sec", "mouth_opening", "brow_height")
FIELDS = ("participant", "trial", "label", "seconds", *FEATURES)
POSE_POINTS = [1, 152, 33, 263, 61, 291]
FACE_MODEL = np.array([
    (0, 0, 0), (0, 330, 65), (-225, -170, 135),
    (225, -170, 135), (-150, 150, 125), (150, 150, 125),
], dtype=np.float64)


class BlinkCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.started = None
        self.closed = None
        self.last = None
        self.events = deque()

    def update(self, now, ratio, threshold):
        if self.last is not None and now - self.last > 0.25:
            self.reset()
        self.last = now
        if self.started is None:
            self.started = now
        if ratio < threshold:
            if self.closed is None:
                self.closed = now
        elif self.closed is not None and ratio >= threshold * 1.15:
            if 0.05 <= now - self.closed <= 0.45:
                self.events.append(now)
            self.closed = None
        while self.events and now - self.events[0] > 15:
            self.events.popleft()
        duration = min(now - self.started, 15)
        return len(self.events) * 60 / duration if duration >= 5 else None


def eye_ratio(points, indices):
    a, b, c, d, e, f = points[indices]
    return (np.linalg.norm(b - f) + np.linalg.norm(c - e)) / max(
        2 * np.linalg.norm(a - d), 1e-6
    )


class Tracker:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=2)
        self.reset()

    def reset(self):
        self.blinks = BlinkCounter()
        self.calibration = []
        self.baseline = None
        self.last_pose = None
        self.last_time = None

    def measure(self, frame, now):
        height, width = frame.shape[:2]
        faces = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks
        if not faces or len(faces) != 1:
            return self.unavailable("Show exactly one face")
        xyz = np.array([(p.x * width, p.y * height, p.z * width)
                        for p in faces[0].landmark])
        points = np.ascontiguousarray(xyz[:, :2])
        low, high = points.min(axis=0), points.max(axis=0)
        if (high - low).min() < 90 or (low < 0).any() or (high >= [width, height]).any():
            return self.unavailable("Move closer and center your face")

        camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1.]])
        ok, rotation, _ = cv2.solvePnP(FACE_MODEL, points[POSE_POINTS], camera, None)
        if not ok:
            return self.unavailable("Head pose unavailable")
        pose = cv2.Rodrigues(rotation)[0]
        if abs(pose[2, 2]) < 0.8:
            return self.unavailable("Face the camera")

        eye = (eye_ratio(points, [33, 160, 158, 133, 153, 144])
               + eye_ratio(points, [362, 385, 387, 263, 373, 380])) / 2
        # Use the mesh's 3D distances and normalize by eye-corner spacing.
        scale = max(np.linalg.norm(xyz[33] - xyz[263]), 1e-6)
        mouth = np.linalg.norm(xyz[13] - xyz[14]) / scale
        brow = (np.linalg.norm(xyz[105] - xyz[159])
                + np.linalg.norm(xyz[334] - xyz[386])) / (2 * scale)
        if self.baseline is None:
            self.calibration.append((now, eye, mouth, brow))
            elapsed = now - self.calibration[0][0]
            if elapsed < 3 or len(self.calibration) < 15:
                return f"Look forward, relax ({min(elapsed, 3):.1f}/3 s)", None, None
            self.baseline = tuple(median(column) for column in zip(*self.calibration))[1:]
            self.calibration.clear()

        blink_rate = self.blinks.update(now, eye, self.baseline[0] * 0.70)
        if self.last_pose is None or now - self.last_time > 0.25:
            head = None
        else:
            relative = self.last_pose.T @ pose
            head = float(np.degrees(np.linalg.norm(cv2.Rodrigues(relative)[0]))) / (now - self.last_time)
        self.last_pose, self.last_time = pose, now
        values = (blink_rate, head, mouth - self.baseline[1], brow - self.baseline[2])
        return "Tracking", values, (tuple(low.astype(int)), tuple(high.astype(int)))

    def unavailable(self, status):
        self.last_pose = None
        self.blinks.reset()
        self.calibration.clear()
        return status, None, None


def run(args, writer=None):
    import tkinter as tk

    from dashboard import Dashboard

    cap = cv2.VideoCapture(args.camera)
    tracker = None
    try:
        if not cap.isOpened():
            raise SystemExit("Cannot open camera. Check permissions or use --camera INDEX.")
        tracker = Tracker()
        root = tk.Tk()
        Dashboard(root, cap, tracker, args, writer)
        root.mainloop()
    finally:
        cap.release()
        if tracker is not None:
            tracker.mesh.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--participant")
    parser.add_argument("--trial")
    parser.add_argument("--label", choices=("truth", "lie"))
    args = parser.parse_args()
    metadata = (args.participant, args.trial, args.label)
    if (args.record and not all(metadata)) or (not args.record and any(metadata)):
        parser.error("Use --record, --participant, --trial and --label together.")
    if args.record:
        try:
            with args.record.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(FIELDS)
                run(args, writer)
        except FileExistsError:
            parser.error("Recording already exists. Choose a new file.")
    else:
        run(args)


if __name__ == "__main__":
    main()
