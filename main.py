import cv2
import time
import math
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import os
import urllib.request
import tempfile

import mediapipe as mp

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class EmotionNeuralNetwork:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.session = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if not ONNX_AVAILABLE:
            return

        model_path = self._get_model_path()

        if os.path.exists(model_path):
            try:
                self.session = ort.InferenceSession(model_path)
                self.is_loaded = True
                return
            except Exception:
                pass

        try:
            self._download_model(model_path)
            self.session = ort.InferenceSession(model_path)
            self.is_loaded = True
        except Exception:
            pass

    def _get_model_path(self) -> str:
        paths = ["emotion_model.onnx", os.path.join(tempfile.gettempdir(), "emotion_model.onnx")]
        for p in paths:
            if os.path.exists(p):
                return p
        return paths[1]

    def _download_model(self, save_path: str):
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_emotion_recognition/emotion-ferplus/emotion-ferplus.onnx"
        urllib.request.urlretrieve(url, save_path)

    def predict(self, face_image: np.ndarray) -> Dict[str, float]:
        if not self.is_loaded or face_image is None or face_image.size == 0:
            return {e: 0.0 for e in self.emotions}

        input_data = self._preprocess(face_image)
        if input_data is None:
            return {e: 0.0 for e in self.emotions}

        try:
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            outputs = self.session.run([output_name], {input_name: input_data})
            predictions = outputs[0][0]

            result = {}
            for i, emotion in enumerate(self.emotions):
                result[emotion] = float(max(0, predictions[i])) if i < len(predictions) else 0.0

            total = sum(result.values())
            if total > 0:
                for key in result:
                    result[key] /= total
            else:
                result['Neutral'] = 1.0

            return result
        except Exception:
            return {e: 0.0 for e in self.emotions}

    def _preprocess(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            resized = cv2.resize(gray, (48, 48))
            normalized = (resized.astype(np.float32) - 127.5) / 127.5
            return np.expand_dims(np.expand_dims(normalized, axis=0), axis=0)
        except Exception:
            return None

    def get_emotion_score(self, emotions: Dict[str, float]) -> float:
        if not emotions:
            return 0.0

        score = 0.0
        score += emotions.get('Fear', 0) * 0.35
        score += emotions.get('Surprise', 0) * 0.20
        score += emotions.get('Disgust', 0) * 0.15

        neutral = emotions.get('Neutral', 1.0)
        if neutral < 0.3:
            score += (1.0 - neutral) * 0.20

        return min(score, 1.0)


class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
            132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        self.RIGHT_EYE = [33, 133, 157, 158, 159, 160, 161, 173]
        self.LEFT_EYE = [362, 263, 387, 386, 385, 384, 398, 466]
        self.RIGHT_IRIS = [468, 469, 470, 471, 472]
        self.LEFT_IRIS = [473, 474, 475, 476, 477]
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        self.NOSE = [1, 2, 3, 4, 5, 6, 197, 195, 5]
        self.EYEBROWS = [46, 53, 52, 65, 55, 285, 282, 283, 293, 300]

        self.blink_history = deque(maxlen=100)
        self.pupil_history = deque(maxlen=50)
        self.head_history = deque(maxlen=50)
        self.mouth_history = deque(maxlen=50)
        self.gaze_history = deque(maxlen=50)
        self.asymmetry_history = deque(maxlen=50)
        self.micro_history = deque(maxlen=50)
        self.jaw_history = deque(maxlen=50)

        self.previous_landmarks = None
        self.face_bbox = None

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {'face_detected': False}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {'face_detected': False}

        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        self.face_bbox = self._get_bbox(landmarks, w, h)

        metrics = {
            'face_detected': True,
            'bbox': self.face_bbox,
            'blink_rate': self._blink_rate(landmarks),
            'pupil_dilation': self._pupil_dilation(landmarks),
            'head_movement': self._head_movement(landmarks),
            'mouth_activity': self._mouth_activity(landmarks),
            'gaze_shift': self._gaze_shift(landmarks),
            'asymmetry': self._asymmetry(landmarks),
            'micro_expressions': self._micro_expressions(landmarks),
            'jaw_tension': self._jaw_tension(landmarks),
            'landmarks': landmarks
        }

        metrics['face_score'] = self._face_score(metrics)
        self.previous_landmarks = landmarks

        return metrics

    def _get_bbox(self, landmarks, w: int, h: int) -> Tuple[int, int, int, int]:
        xs, ys = [], []
        for idx in self.FACE_OVAL:
            if idx < len(landmarks.landmark):
                p = landmarks.landmark[idx]
                xs.append(p.x * w)
                ys.append(p.y * h)

        if not xs or not ys:
            return (0, 0, w, h)

        x1 = max(0, int(min(xs)) - 20)
        y1 = max(0, int(min(ys)) - 30)
        x2 = min(w, int(max(xs)) + 20)
        y2 = min(h, int(max(ys)) + 30)
        return (x1, y1, x2, y2)

    def _blink_rate(self, landmarks) -> float:
        re = self._points(landmarks, self.RIGHT_EYE)
        le = self._points(landmarks, self.LEFT_EYE)
        if len(re) < 6 or len(le) < 6:
            return 0.0

        ear = (self._ear(re) + self._ear(le)) / 2.0
        is_blink = ear < 0.2
        self.blink_history.append(1 if is_blink else 0)

        if len(self.blink_history) > 20:
            rate = sum(list(self.blink_history)[-20:]) / 20 * 15
            return min(rate, 1.0)
        return 0.0

    def _pupil_dilation(self, landmarks) -> float:
        ri = self._points(landmarks, self.RIGHT_IRIS)
        li = self._points(landmarks, self.LEFT_IRIS)
        if len(ri) < 3 or len(li) < 3:
            return 0.0

        rr = self._radius(ri)
        lr = self._radius(li)

        re = self._points(landmarks, self.RIGHT_EYE)
        le = self._points(landmarks, self.LEFT_EYE)
        if len(re) < 6 or len(le) < 6:
            return 0.0

        rw = self._dist(re[0], re[3])
        lw = self._dist(le[0], le[3])
        if rw + lw < 0.001:
            return 0.0

        ratio = (rr + lr) / (rw + lw)
        self.pupil_history.append(ratio)

        if len(self.pupil_history) > 20:
            recent = list(self.pupil_history)[-10:]
            prev = list(self.pupil_history)[-20:-10]
            if prev and np.mean(prev) > 0:
                dilation = (np.mean(recent) - np.mean(prev)) / np.mean(prev)
                if dilation > 0.05:
                    return min(dilation * 3, 1.0)
        return 0.0

    def _head_movement(self, landmarks) -> float:
        nose = self._points(landmarks, self.NOSE)
        if len(nose) < 5:
            return 0.0

        center = np.mean(nose, axis=0)
        if self.previous_landmarks is not None:
            prev = self._points(self.previous_landmarks, self.NOSE)
            if len(prev) > 5:
                prev_center = np.mean(prev, axis=0)
                movement = self._dist(center, prev_center)
                self.head_history.append(movement)

        if len(self.head_history) > 30:
            avg = np.mean(list(self.head_history)[-20:])
            return min(avg * 50, 1.0)
        return 0.0

    def _mouth_activity(self, landmarks) -> float:
        lips = self._points(landmarks, self.LIPS)
        if len(lips) < 10:
            return 0.0

        h = self._dist(lips[0], lips[3])
        w = self._dist(lips[2], lips[4])
        if w < 0.001:
            return 0.0

        ratio = h / w
        self.mouth_history.append(ratio)

        if len(self.mouth_history) > 20:
            return min(np.std(list(self.mouth_history)) * 10, 1.0)
        return 0.0

    def _gaze_shift(self, landmarks) -> float:
        ri = self._points(landmarks, self.RIGHT_IRIS)
        li = self._points(landmarks, self.LEFT_IRIS)
        if len(ri) < 3 or len(li) < 3:
            return 0.0

        rc = np.mean(ri, axis=0)
        lc = np.mean(li, axis=0)

        re = self._points(landmarks, self.RIGHT_EYE)
        le = self._points(landmarks, self.LEFT_EYE)
        if len(re) < 6 or len(le) < 6:
            return 0.0

        rec = np.mean(re, axis=0)
        lec = np.mean(le, axis=0)

        ro = self._dist(rc, rec)
        lo = self._dist(lc, lec)
        avg = (ro + lo) / 2.0
        self.gaze_history.append(avg)

        if len(self.gaze_history) > 20:
            return min(np.std(list(self.gaze_history)) * 20, 1.0)
        return 0.0

    def _asymmetry(self, landmarks) -> float:
        left = self._points(landmarks, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        right = self._points(landmarks, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])

        if len(left) < 5 or len(right) < 5:
            return 0.0

        lm = np.mean(left, axis=0)
        rm = np.mean(right, axis=0)
        asym = abs(lm[0] - rm[0]) + abs(lm[1] - rm[1])
        self.asymmetry_history.append(asym)

        if len(self.asymmetry_history) > 20:
            avg = np.mean(list(self.asymmetry_history))
            return min(max(0, (avg - 0.01) * 50), 1.0)
        return 0.0

    def _micro_expressions(self, landmarks) -> float:
        if self.previous_landmarks is None:
            return 0.0

        total, count = 0.0, 0
        for idx in self.RIGHT_EYE + self.LEFT_EYE + self.LIPS + self.EYEBROWS:
            if idx < len(landmarks.landmark) and idx < len(self.previous_landmarks.landmark):
                c = landmarks.landmark[idx]
                p = self.previous_landmarks.landmark[idx]
                total += math.sqrt((c.x - p.x) ** 2 + (c.y - p.y) ** 2)
                count += 1

        if count == 0:
            return 0.0

        avg = total / count
        self.micro_history.append(avg)

        if len(self.micro_history) > 20:
            return min(np.std(list(self.micro_history)) * 50, 1.0)
        return 0.0

    def _jaw_tension(self, landmarks) -> float:
        lips = self._points(landmarks, self.LIPS)
        if len(lips) < 10:
            return 0.0

        top = np.mean(lips[:10], axis=0)
        bottom = np.mean(lips[10:], axis=0)
        dist = self._dist(top, bottom)
        self.jaw_history.append(dist)

        if len(self.jaw_history) > 20:
            avg = np.mean(list(self.jaw_history)[-20:])
            if avg < 0.02:
                return min((0.02 - avg) * 50, 1.0)
        return 0.0

    def _face_score(self, metrics: Dict) -> float:
        weights = {
            'blink_rate': 0.15,
            'pupil_dilation': 0.20,
            'head_movement': 0.10,
            'mouth_activity': 0.12,
            'gaze_shift': 0.15,
            'asymmetry': 0.08,
            'micro_expressions': 0.12,
            'jaw_tension': 0.08
        }
        score, total = 0.0, 0.0
        for key, w in weights.items():
            if key in metrics:
                score += metrics[key] * w
                total += w
        return score / total if total > 0 else 0.0

    def _points(self, landmarks, indices):
        pts = []
        for idx in indices:
            if idx < len(landmarks.landmark):
                p = landmarks.landmark[idx]
                pts.append([p.x, p.y])
        return np.array(pts)

    def _ear(self, eye):
        if len(eye) < 6:
            return 0.0
        a = self._dist(eye[1], eye[5])
        b = self._dist(eye[2], eye[4])
        c = self._dist(eye[0], eye[3])
        return (a + b) / (2.0 * c + 0.001)

    def _radius(self, pts):
        if len(pts) < 3:
            return 0.0
        c = np.mean(pts, axis=0)
        return np.mean([self._dist(p, c) for p in pts])

    def _dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class LieDetectorApp:
    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.emotion_nn = EmotionNeuralNetwork()

        self.width = 1280
        self.height = 720

        self.is_running = False
        self.frame_count = 0
        self.fps = 0.0
        self.start_time = None

        self.results = {}
        self.score_history = deque(maxlen=100)
        self.screenshot_count = 0
        self.alert_count = 0

    def start(self):
        self.is_running = True
        self.start_time = time.time()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            return

        cv2.namedWindow('Lie Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Lie Detector', self.width, self.height)

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                continue

            self.frame_count += 1

            results = self._analyze(frame)
            self.results = results

            display = self._render(frame, results)

            if results.get('level') == 'HIGH':
                display = self._render_alert(display)
                self.alert_count += 1

            cv2.imshow('Lie Detector', display)

            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.is_running = False
            elif key == ord('s'):
                self._screenshot(display)
            elif key == ord('r'):
                self._reset()
            elif key == ord('f'):
                self._toggle_fullscreen()

        cap.release()
        cv2.destroyAllWindows()

    def _analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        results = {
            'timestamp': datetime.now().isoformat(),
            'face_detected': False
        }

        face_data = self.face_analyzer.analyze(frame)

        if face_data.get('face_detected', False):
            results['face_detected'] = True
            results['bbox'] = face_data.get('bbox')

            bbox = face_data.get('bbox')
            if bbox:
                x1, y1, x2, y2 = bbox
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    emotions = self.emotion_nn.predict(face_roi)
                    results['emotions'] = emotions
                    results['nn_score'] = self.emotion_nn.get_emotion_score(emotions)

            face_score = face_data.get('face_score', 0)
            nn_score = results.get('nn_score', 0)

            composite = face_score * 0.5 + nn_score * 0.5
            if face_score > 0.3 and nn_score > 0.3:
                composite = min(1.0, composite * 1.15)

            results['score'] = composite
            results['face_score'] = face_score
            results['composite_score'] = composite
            results['face_data'] = face_data

            if composite > 0.55:
                results['level'] = 'HIGH'
            elif composite > 0.35:
                results['level'] = 'MEDIUM'
            elif composite > 0.15:
                results['level'] = 'LOW'
            else:
                results['level'] = 'NONE'

            results['indicators'] = self._get_indicators(face_data)
            self.score_history.append(composite)

        return results

    def _get_indicators(self, face_data: Dict) -> List[str]:
        indicators = []
        if face_data.get('blink_rate', 0) > 0.3:
            indicators.append('Frequent blinking')
        if face_data.get('pupil_dilation', 0) > 0.3:
            indicators.append('Pupil dilation')
        if face_data.get('gaze_shift', 0) > 0.3:
            indicators.append('Gaze shift')
        if face_data.get('micro_expressions', 0) > 0.3:
            indicators.append('Micro expressions')
        if face_data.get('asymmetry', 0) > 0.3:
            indicators.append('Facial asymmetry')
        if face_data.get('jaw_tension', 0) > 0.3:
            indicators.append('Jaw tension')
        if face_data.get('head_movement', 0) > 0.3:
            indicators.append('Head movement')
        if face_data.get('mouth_activity', 0) > 0.3:
            indicators.append('Mouth activity')
        return indicators

    def _render(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        display = frame.copy()
        h, w = display.shape[:2]

        if results.get('face_detected', False):
            bbox = results.get('bbox')
            if bbox:
                x1, y1, x2, y2 = bbox
                score = results.get('score', 0)

                if score > 0.55:
                    color = (0, 0, 255)
                elif score > 0.35:
                    color = (0, 165, 255)
                elif score > 0.15:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                c = 15
                for x, y in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
                    cv2.line(display, (x - c, y), (x + c, y), color, 2)
                    cv2.line(display, (x, y - c), (x, y + c), color, 2)

                cv2.putText(display, f"{score*100:.0f}%", (x1, y1 - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        px, py, pw, ph = 10, 10, 380, 480
        overlay = display.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        cv2.rectangle(display, (px, py), (px + pw, py + ph), (0, 200, 0), 2)

        y = py + 25

        cv2.putText(display, "LIE DETECTOR", (px + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y += 30

        status = "FACE" if results.get('face_detected') else "NO FACE"
        cv2.putText(display, status, (px + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if results.get('face_detected') else (0, 0, 255), 1)
        y += 25

        nn_status = "NN: OK" if self.emotion_nn.is_loaded else "NN: FALLBACK"
        cv2.putText(display, nn_status, (px + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        y += 25

        cv2.putText(display, f"FPS: {self.fps:.1f}", (px + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 30

        cv2.line(display, (px + 10, y), (px + pw - 10, y), (80, 80, 80), 1)
        y += 20

        score = results.get('score', 0)
        if score > 0.55:
            color, level = (0, 0, 255), "HIGH"
        elif score > 0.35:
            color, level = (0, 165, 255), "MEDIUM"
        elif score > 0.15:
            color, level = (0, 255, 255), "LOW"
        else:
            color, level = (0, 255, 0), "NONE"

        cv2.putText(display, "DECEPTION SCORE", (px + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28
        cv2.putText(display, f"{score*100:.1f}%", (px + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
        y += 45
        cv2.putText(display, f"LEVEL: {level}", (px + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 30

        bx, by = px + 10, y
        bw, bh = pw - 20, 16
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
        cv2.rectangle(display, (bx, by), (bx + int(bw * score), by + bh), color, -1)
        y += 30

        cv2.line(display, (px + 10, y), (px + pw - 10, y), (80, 80, 80), 1)
        y += 20

        cv2.putText(display, "NEURAL NETWORK EMOTIONS", (px + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        y += 22

        emotions = results.get('emotions', {})
        if emotions:
            sorted_em = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:4]
            for name, prob in sorted_em:
                if prob > 0.05:
                    bar_w = min(int(prob * 100), 120)
                    cv2.rectangle(display, (px + 15, y - 3), (px + 15 + bar_w, y + 12), (100, 200, 255), -1)
                    cv2.putText(display, f"{name}: {prob*100:.0f}%", (px + 145, y + 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y += 20
        else:
            cv2.putText(display, "Waiting...", (px + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y += 20

        y += 5
        cv2.line(display, (px + 10, y), (px + pw - 10, y), (80, 80, 80), 1)
        y += 20

        cv2.putText(display, "INDICATORS", (px + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        y += 22

        indicators = results.get('indicators', [])
        if indicators:
            for ind in indicators[:5]:
                cv2.putText(display, f"  {ind}", (px + 15, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 180, 180), 1)
                y += 18
        else:
            cv2.putText(display, "No indicators", (px + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 150, 100), 1)

        cv2.putText(display, "ESC/q:Quit | s:Shot | r:Reset | f:Fullscreen", (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return display

    def _render_alert(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 0, 255), 4)

        text = "DECEPTION DETECTED"
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        x = (w - size[0]) // 2

        cv2.putText(frame, text, (x + 2, 47), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(frame, text, (x, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, f"Alert #{self.alert_count}", (w - 150, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def _screenshot(self, frame):
        self.screenshot_count += 1
        cv2.imwrite(f'screenshot_{self.screenshot_count}.png', frame)

    def _reset(self):
        self.score_history.clear()
        self.alert_count = 0

    def _toggle_fullscreen(self):
        fs = cv2.getWindowProperty('Lie Detector', cv2.WND_PROP_FULLSCREEN)
        if fs == 1:
            cv2.setWindowProperty('Lie Detector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Lie Detector', self.width, self.height)
        else:
            cv2.setWindowProperty('Lie Detector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def main():
    try:
        app = LieDetectorApp()
        app.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()