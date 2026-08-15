"""
Audio analysis module for voice stress detection.
Extracts acoustic features indicative of deception.
"""

import numpy as np
import librosa
from collections import deque
from typing import Dict, Optional, Any

class AudioAnalyzer:
    """
    Analyzes audio signals and extracts acoustic features.
    """
    
    def __init__(self):
        """Initialize the audio analyzer with feature buffers."""
        self.pitch_history = deque(maxlen=50)
        self.energy_history = deque(maxlen=50)
        self.speech_rate_history = deque(maxlen=10)
        self.voice_quality_history = deque(maxlen=30)
        
    def analyze(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Analyze audio signal and extract features.
        
        Args:
            audio_data: Audio signal array.
            sample_rate: Sample rate in Hz.
            
        Returns:
            Dictionary containing audio metrics.
        """
        if len(audio_data) < sample_rate * 0.1:
            return {}
        
        try:
            metrics = {
                'pitch_variation': self._extract_pitch_variation(audio_data, sample_rate),
                'energy_level': self._extract_energy_level(audio_data, sample_rate),
                'speech_rate': self._extract_speech_rate(audio_data, sample_rate),
                'voice_quality': self._extract_voice_quality(audio_data, sample_rate),
                'stress_index': 0.0
            }
            
            # Calculate composite stress index
            metrics['stress_index'] = (
                metrics['pitch_variation'] * 0.3 +
                metrics['energy_level'] * 0.2 +
                metrics['speech_rate'] * 0.3 +
                metrics['voice_quality'] * 0.2
            )
            
            return metrics
            
        except Exception as error:
            print(f"Audio analysis error: {error}")
            return {}
    
    def _extract_pitch_variation(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Extract pitch variation from audio signal.
        
        Args:
            audio: Audio signal array.
            sample_rate: Sample rate in Hz.
            
        Returns:
            Normalized pitch variation (0.0 to 1.0).
        """
        try:
            pitches = librosa.piptrack(
                y=audio, 
                sr=sample_rate, 
                fmin=50, 
                fmax=400
            )
            
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = pitches[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) < 10:
                return 0.0
            
            mean_pitch = np.mean(pitch_values)
            if mean_pitch < 0.001:
                return 0.0
            
            variation = np.std(pitch_values) / mean_pitch
            self.pitch_history.append(variation)
            
            return min(variation * 5.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _extract_energy_level(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Extract energy level from audio signal.
        
        Args:
            audio: Audio signal array.
            sample_rate: Sample rate in Hz.
            
        Returns:
            Normalized energy variation (0.0 to 1.0).
        """
        try:
            rms = np.sqrt(np.mean(audio**2))
            
            if rms < 1e-10:
                return 0.0
            
            energy_db = 20 * np.log10(rms)
            normalized = (energy_db + 60) / 60
            normalized = np.clip(normalized, 0.0, 1.0)
            
            self.energy_history.append(normalized)
            
            if len(self.energy_history) >= 10:
                energy_std = np.std(list(self.energy_history))
                return min(energy_std * 3.0, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _extract_speech_rate(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Extract speech rate from audio signal.
        
        Args:
            audio: Audio signal array.
            sample_rate: Sample rate in Hz.
            
        Returns:
            Normalized speech rate variation (0.0 to 1.0).
        """
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            energy = np.sum(np.abs(mfcc), axis=0)
            
            if np.max(energy) < 0.001:
                return 0.0
            
            energy_normalized = energy / np.max(energy)
            threshold = 0.1
            speech_segments = energy_normalized > threshold
            
            changes = np.sum(np.diff(speech_segments.astype(int)) != 0)
            speech_rate = changes / (len(energy_normalized) + 0.001)
            
            self.speech_rate_history.append(speech_rate)
            
            if len(self.speech_rate_history) >= 5:
                rate_std = np.std(list(self.speech_rate_history))
                return min(rate_std * 10.0, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _extract_voice_quality(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Extract voice quality features from audio signal.
        
        Args:
            audio: Audio signal array.
            sample_rate: Sample rate in Hz.
            
        Returns:
            Normalized voice quality variation (0.0 to 1.0).
        """
        try:
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, 
                sr=sample_rate
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, 
                sr=sample_rate
            )
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, 
                sr=sample_rate
            )
            
            centroid_std = np.std(spectral_centroids)
            rolloff_std = np.std(spectral_rolloff)
            bandwidth_std = np.std(spectral_bandwidth)
            
            mean_centroid = np.mean(spectral_centroids)
            if mean_centroid < 0.001:
                return 0.0
            
            quality_score = (centroid_std + rolloff_std + bandwidth_std) / mean_centroid
            
            self.voice_quality_history.append(quality_score)
            
            return min(quality_score * 10.0, 1.0)
            
        except Exception:
            return 0.0