"""
Face analysis module using MediaPipe Face Mesh.
Extracts facial metrics for deception detection.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
from typing import Dict, List, Any

class FaceAnalyzer:
    """
    Analyzes facial features and extracts metrics indicative of deception.
    Uses MediaPipe Face Mesh for landmark detection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the face analyzer with MediaPipe Face Mesh.
        
        Args:
            config: Configuration dictionary with analysis parameters.
        """
        # Default configuration
        if config is None:
            config = {
                'detection_confidence': 0.5,
                'tracking_confidence': 0.5,
                'max_faces': 1,
                'refine_landmarks': True
            }
        
        # Initialize MediaPipe Face Mesh for version 0.10.13
        try:
            # Try to import from mediapipe.python.solutions
            from mediapipe.python.solutions import face_mesh
            self.face_mesh = face_mesh.FaceMesh(
                max_num_faces=config.get('max_faces', 1),
                refine_landmarks=config.get('refine_landmarks', True),
                min_detection_confidence=config.get('detection_confidence', 0.5),
                min_tracking_confidence=config.get('tracking_confidence', 0.5)
            )
        except ImportError:
            # Fallback to direct import
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=config.get('max_faces', 1),
                refine_landmarks=config.get('refine_landmarks', True),
                min_detection_confidence=config.get('detection_confidence', 0.5),
                min_tracking_confidence=config.get('tracking_confidence', 0.5)
            )
        
        # Landmark indices for facial features
        self.RIGHT_EYE_INDICES = [33, 133, 157, 158, 159, 160, 161, 173]
        self.RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]
        self.LEFT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 466]
        self.LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]
        self.LIPS_INDICES = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
        ]
        self.NOSE_INDICES = [1, 2, 3, 4, 5, 6, 197, 195, 5]
        
        # Historical buffers for smoothing
        self.blink_history = deque(maxlen=30)
        self.pupil_history = deque(maxlen=10)
        self.head_history = deque(maxlen=30)
        self.mouth_history = deque(maxlen=20)
        self.gaze_history = deque(maxlen=20)
        
        # Previous state for comparison
        self.previous_landmarks = None
        
    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a single frame and extract facial metrics.
        
        Args:
            frame: BGR image from camera.
            
        Returns:
            Dictionary containing facial metrics.
        """
        if frame is None or frame.size == 0:
            return {'face_present': False}
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {'face_present': False}
        
        landmarks = results.multi_face_landmarks[0]
        
        metrics = {
            'blink_rate': self._calculate_blink_rate(landmarks),
            'pupil_dilation': self._calculate_pupil_dilation(landmarks),
            'mouth_activity': self._calculate_mouth_activity(landmarks),
            'head_movement': self._calculate_head_movement(landmarks),
            'gaze_shift': self._calculate_gaze_shift(landmarks),
            'face_present': True,
            'landmarks': landmarks
        }
        
        self.previous_landmarks = landmarks
        
        return metrics
    
    def _calculate_blink_rate(self, landmarks) -> float:
        """
        Calculate blink rate using Eye Aspect Ratio (EAR).
        
        Args:
            landmarks: MediaPipe face landmarks.
            
        Returns:
            Normalized blink rate (0.0 to 1.0).
        """
        right_eye = self._get_landmark_points(landmarks, self.RIGHT_EYE_INDICES)
        left_eye = self._get_landmark_points(landmarks, self.LEFT_EYE_INDICES)
        
        if len(right_eye) < 6 or len(left_eye) < 6:
            return 0.0
        
        ear_right = self._calculate_ear(right_eye)
        ear_left = self._calculate_ear(left_eye)
        ear_average = (ear_right + ear_left) / 2.0
        
        is_blink = ear_average < 0.25
        self.blink_history.append(1 if is_blink else 0)
        
        if len(self.blink_history) >= 10:
            blink_count = sum(self.blink_history)
            blink_rate = blink_count / len(self.blink_history) * 10.0
            return min(blink_rate, 1.0)
        
        return 0.0
    
    def _calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio for a single eye.
        
        Args:
            eye_points: Array of eye landmark coordinates.
            
        Returns:
            EAR value.
        """
        if len(eye_points) < 6:
            return 0.0
        
        p2_p6 = self._calculate_distance(eye_points[1], eye_points[5])
        p3_p5 = self._calculate_distance(eye_points[2], eye_points[4])
        p1_p4 = self._calculate_distance(eye_points[0], eye_points[3])
        
        if p1_p4 < 0.001:
            return 0.0
        
        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        return ear
    
    def _calculate_pupil_dilation(self, landmarks) -> float:
        """
        Calculate pupil dilation based on relative iris size.
        
        Args:
            landmarks: MediaPipe face landmarks.
            
        Returns:
            Normalized pupil dilation (0.0 to 1.0).
        """
        right_iris = self._get_landmark_points(landmarks, self.RIGHT_IRIS_INDICES)
        left_iris = self._get_landmark_points(landmarks, self.LEFT_IRIS_INDICES)
        
        if len(right_iris) < 3 or len(left_iris) < 3:
            return 0.0
        
        right_radius = self._calculate_radius(right_iris)
        left_radius = self._calculate_radius(left_iris)
        
        right_eye = self._get_landmark_points(landmarks, self.RIGHT_EYE_INDICES)
        left_eye = self._get_landmark_points(landmarks, self.LEFT_EYE_INDICES)
        
        if len(right_eye) < 6 or len(left_eye) < 6:
            return 0.0
        
        right_eye_width = self._calculate_distance(right_eye[0], right_eye[3])
        left_eye_width = self._calculate_distance(left_eye[0], left_eye[3])
        
        if right_eye_width + left_eye_width < 0.001:
            return 0.0
        
        pupil_ratio = (right_radius + left_radius) / (right_eye_width + left_eye_width)
        self.pupil_history.append(pupil_ratio)
        
        if len(self.pupil_history) >= 10:
            baseline = np.mean(list(self.pupil_history)[:5])
            current = np.mean(list(self.pupil_history)[5:])
            
            if baseline < 0.001:
                return 0.0
            
            dilation = (current - baseline) / baseline
            return max(0.0, min(dilation * 5.0, 1.0))
        
        return 0.0
    
    def _calculate_mouth_activity(self, landmarks) -> float:
        """
        Calculate mouth activity based on lip movement.
        
        Args:
            landmarks: MediaPipe face landmarks.
            
        Returns:
            Normalized mouth activity (0.0 to 1.0).
        """
        lips = self._get_landmark_points(landmarks, self.LIPS_INDICES)
        
        if len(lips) < 10:
            return 0.0
        
        mouth_height = self._calculate_distance(lips[0], lips[3])
        mouth_width = self._calculate_distance(lips[2], lips[4])
        
        if mouth_width < 0.001:
            return 0.0
        
        ratio = mouth_height / mouth_width
        self.mouth_history.append(ratio)
        
        if len(self.mouth_history) >= 10:
            std_dev = np.std(list(self.mouth_history))
            return min(std_dev * 10.0, 1.0)
        
        return 0.0
    
    def _calculate_head_movement(self, landmarks) -> float:
        """
        Calculate head movement based on nose position changes.
        
        Args:
            landmarks: MediaPipe face landmarks.
            
        Returns:
            Normalized head movement (0.0 to 1.0).
        """
        nose = self._get_landmark_points(landmarks, self.NOSE_INDICES)
        
        if len(nose) < 5:
            return 0.0
        
        nose_center = np.mean(nose, axis=0)
        
        if self.previous_landmarks is not None:
            previous_nose = self._get_landmark_points(
                self.previous_landmarks, 
                self.NOSE_INDICES
            )
            
            if len(previous_nose) > 0:
                previous_center = np.mean(previous_nose, axis=0)
                movement = self._calculate_distance(nose_center, previous_center)
                self.head_history.append(movement)
        
        if len(self.head_history) >= 10:
            average_movement = np.mean(list(self.head_history))
            return min(average_movement * 5.0, 1.0)
        
        return 0.0
    
    def _calculate_gaze_shift(self, landmarks) -> float:
        """
        Calculate gaze shift based on iris position changes.
        
        Args:
            landmarks: MediaPipe face landmarks.
            
        Returns:
            Normalized gaze shift (0.0 to 1.0).
        """
        right_iris = self._get_landmark_points(landmarks, self.RIGHT_IRIS_INDICES)
        left_iris = self._get_landmark_points(landmarks, self.LEFT_IRIS_INDICES)
        
        if len(right_iris) < 3 or len(left_iris) < 3:
            return 0.0
        
        right_center = np.mean(right_iris, axis=0)
        left_center = np.mean(left_iris, axis=0)
        
        right_eye = self._get_landmark_points(landmarks, self.RIGHT_EYE_INDICES)
        left_eye = self._get_landmark_points(landmarks, self.LEFT_EYE_INDICES)
        
        if len(right_eye) < 6 or len(left_eye) < 6:
            return 0.0
        
        right_eye_center = np.mean(right_eye, axis=0)
        left_eye_center = np.mean(left_eye, axis=0)
        
        right_offset = self._calculate_distance(right_center, right_eye_center)
        left_offset = self._calculate_distance(left_center, left_eye_center)
        
        average_offset = (right_offset + left_offset) / 2.0
        self.gaze_history.append(average_offset)
        
        if len(self.gaze_history) >= 10:
            std_dev = np.std(list(self.gaze_history))
            return min(std_dev * 5.0, 1.0)
        
        return 0.0
    
    def _get_landmark_points(self, landmarks, indices: List[int]) -> np.ndarray:
        """
        Extract landmark coordinates by indices.
        
        Args:
            landmarks: MediaPipe face landmarks.
            indices: List of landmark indices.
            
        Returns:
            Array of [x, y] coordinates.
        """
        points = []
        
        for idx in indices:
            if idx < len(landmarks.landmark):
                point = landmarks.landmark[idx]
                points.append([point.x, point.y])
        
        return np.array(points)
    
    def _calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            p1: First point coordinates.
            p2: Second point coordinates.
            
        Returns:
            Euclidean distance.
        """
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _calculate_radius(self, points: np.ndarray) -> float:
        """
        Calculate radius of a set of points.
        
        Args:
            points: Array of point coordinates.
            
        Returns:
            Average radius from centroid.
        """
        if len(points) < 3:
            return 0.0
        
        center = np.mean(points, axis=0)
        distances = [self._calculate_distance(p, center) for p in points]
        return np.mean(distances)
    
    def visualize_landmarks(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """
        Visualize facial landmarks on the frame.
        
        Args:
            frame: Input image.
            landmarks: MediaPipe face landmarks.
            
        Returns:
            Frame with visualized landmarks.
        """
        if landmarks is None:
            return frame
        
        # Draw eye landmarks
        for idx in self.RIGHT_EYE_INDICES + self.LEFT_EYE_INDICES:
            if idx < len(landmarks.landmark):
                point = landmarks.landmark[idx]
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Draw iris landmarks
        for idx in self.RIGHT_IRIS_INDICES + self.LEFT_IRIS_INDICES:
            if idx < len(landmarks.landmark):
                point = landmarks.landmark[idx]
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
        return frame