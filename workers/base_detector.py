"""
Base detector class for GPU-accelerated ensemble detection.
Provides common functionality for all detector implementations:
- CUDA device management
- Mixed precision (FP16) inference
- Batch processing support
- Model caching and preallocation
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.cuda.amp import autocast

from config.settings import (
    USE_GPU,
    MIXED_PRECISION_FP16,
    MODEL_CACHE_DIR,
    DETECTION_CONFIDENCE_THRESHOLD,
)


logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """
    Abstract base class for detection models.
    
    Features:
    - Automatic GPU/CPU device selection
    - FP16 mixed precision for faster inference
    - Batch processing with configurable batch size
    - Model caching in local directory
    - Detection output standardization
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        batch_size: int = 1,
        confidence_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
        use_fp16: bool = MIXED_PRECISION_FP16,
    ):
        """
        Initialize detector.
        
        Args:
            model_name: Unique identifier for this detector (e.g., "yolo_vit", "frcnn_rtdetr")
            device: Device string ("cuda:0", "cuda:1", "cpu")
            batch_size: Number of frames to process in parallel
            confidence_threshold: Minimum confidence for detections
            use_fp16: Enable mixed precision (FP16) inference
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.use_fp16 = use_fp16 and MIXED_PRECISION_FP16

        # Device management
        self.device = self._get_device(device)
        logger.info(
            f"Detector '{model_name}' initialized on device: {self.device} "
            f"(FP16: {self.use_fp16}, Batch Size: {batch_size})"
        )

        # Create model cache directory
        Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)

        # Initialize model (implemented by subclasses)
        self.model = None
        self.model_info = {}

    def _get_device(self, requested_device: str) -> torch.device:
        """
        Get appropriate PyTorch device.
        Falls back to CPU if CUDA not available.
        """
        if not USE_GPU or requested_device == "cpu":
            return torch.device("cpu")

        try:
            device = torch.device(requested_device)
            # Verify CUDA device is accessible
            torch.cuda.current_device()
            logger.info(
                f"Using GPU device: {requested_device} "
                f"(GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB)"
            )
            return device
        except (RuntimeError, AssertionError):
            logger.warning(
                f"Requested device '{requested_device}' not available. Falling back to CPU."
            )
            return torch.device("cpu")

    def _prealloc_gpu_memory(self, expected_size_mb: int = 500) -> None:
        """
        Pre-allocate GPU memory to avoid fragmentation during inference.
        Only relevant when using GPU.
        """
        if str(self.device).startswith("cuda"):
            try:
                _ = torch.empty(expected_size_mb * 1024 * 1024 // 4, dtype=torch.float32, device=self.device)
                logger.info(f"Pre-allocated {expected_size_mb} MB on {self.device}")
            except RuntimeError as e:
                logger.warning(f"Could not pre-allocate GPU memory: {e}")

    def _set_inference_mode(self) -> None:
        """Set model to evaluation mode and disable gradients."""
        if self.model is not None:
            self.model.eval()
            torch.set_grad_enabled(False)

    @abstractmethod
    def initialize_model(self) -> None:
        """
        Load and configure the detection model.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Run inference on a single frame.
        
        Args:
            frame: Input frame as numpy array (H, W, 3) in BGR format
            
        Returns:
            Dictionary with:
            {
                'boxes': np.ndarray of shape (N, 4) in (x1, y1, x2, y2) format,
                'confidences': np.ndarray of shape (N,),
                'class_ids': np.ndarray of shape (N,),
                'execution_time_ms': float,
                'device': str,
            }
        """
        pass

    @abstractmethod
    def detect_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Run inference on a batch of frames (optional optimization).
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            List of detection dictionaries
        """
        pass

    def validate_detections(self, detections: Dict) -> bool:
        """
        Validate detection output format.
        """
        required_keys = {"boxes", "confidences", "class_ids", "execution_time_ms", "device"}
        if not all(k in detections for k in required_keys):
            logger.warning(f"Missing required keys in detections: {detections.keys()}")
            return False

        if len(detections["boxes"]) != len(detections["confidences"]):
            logger.warning("Mismatch between boxes and confidences length")
            return False

        return True

    def postprocess_detections(
        self, 
        boxes: np.ndarray, 
        confidences: np.ndarray, 
        class_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Apply confidence threshold and return valid detections.
        """
        mask = confidences >= self.confidence_threshold
        return (
            boxes[mask],
            confidences[mask],
            class_ids[mask] if class_ids is not None else None,
        )

    def get_model_info(self) -> Dict:
        """Return metadata about the model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "use_fp16": self.use_fp16,
            "confidence_threshold": self.confidence_threshold,
            **self.model_info,
        }

    def cleanup(self) -> None:
        """Release GPU memory and clean up resources."""
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        logger.info(f"Detector '{self.model_name}' cleaned up")
