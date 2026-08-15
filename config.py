"""
Конфигурация системы
"""
import os

class Config:
    # Камера
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    
    # Модели
    EMOTION_MODEL_PATH = "emotion_model.onnx"
    
    # Пороги детекции
    DECEPTION_THRESHOLD_HIGH = 0.55
    DECEPTION_THRESHOLD_MEDIUM = 0.35
    DECEPTION_THRESHOLD_LOW = 0.15
    
    # Веса для композитного скора
    WEIGHTS = {
        'face': 0.35,
        'voice': 0.25,
        'body': 0.15,
        'psych': 0.25
    }