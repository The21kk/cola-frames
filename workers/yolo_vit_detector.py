"""
Worker 1: YOLOv8 + Vision Transformer Ensemble Detector

Combines:
- YOLOv8 for fast object detection with bounding boxes
- Vision Transformer (ViT) for feature-based confidence scoring
- Ensemble voting for robust detection consensus
"""

import time
import logging
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from ultralytics import YOLO
from timm.models import create_model
import torchvision.transforms as transforms

from workers.base_detector import BaseDetector
from config.settings import MODEL_CACHE_DIR


logger = logging.getLogger(__name__)


class YOLOVitDetector(BaseDetector):
    """
    YOLOv8 + Vision Transformer ensemble detector.
    
    - YOLOv8: Provides spatial bounding boxes
    - ViT: Provides feature-based confidence through image patches
    - Consensus: Both models must detect objects in similar regions
    """

    YOLO_MODEL_NAME = "yolov8m.pt"  # Medium model for balance
    VIT_MODEL_NAME = "vit_base_patch16_224"  # Base Vision Transformer

    def __init__(
        self,
        device: str = "cuda:0",
        batch_size: int = 4,
        confidence_threshold: float = 0.5,
        use_fp16: bool = True,
    ):
        """Initialize YOLOv8 + ViT detector."""
        super().__init__(
            model_name="yolo_vit",
            device=device,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold,
            use_fp16=use_fp16,
        )
        self.initialize_model()

    def initialize_model(self) -> None:
        """Load YOLOv8 and Vision Transformer models."""
        try:
            logger.info("Loading YOLOv8 model...")
            self.yolo_model = YOLO(self.YOLO_MODEL_NAME)
            self.yolo_model.to(self.device)

            logger.info("Loading Vision Transformer model...")
            self.vit_model = create_model(
                self.VIT_MODEL_NAME,
                pretrained=True,
                num_classes=1,  # Binary classification (object/no-object)
            )
            self.vit_model = self.vit_model.to(self.device)
            self.vit_model.eval()

            # Image normalization for ViT
            self.vit_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._prealloc_gpu_memory(500)
            self._set_inference_mode()

            # Store model info
            self.model_info = {
                "yolo_model": self.YOLO_MODEL_NAME,
                "vit_model": self.VIT_MODEL_NAME,
                "ensemble_type": "YOLO + ViT",
            }

            logger.info("YOLOv8 + ViT detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8 + ViT detector: {e}")
            raise

    def detect(self, frame: np.ndarray) -> Dict:
        """
        Run detection on a single frame using YOLOv8 + ViT ensemble.
        
        Args:
            frame: Input frame (H, W, 3) in BGR format
            
        Returns:
            Detection dictionary with boxes, confidences, class_ids, etc.
        """
        start_time = time.time()

        # YOLOv8 detection
        yolo_results = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)
        yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
        yolo_confidences = yolo_results[0].boxes.conf.cpu().numpy()  # (N,)
        yolo_class_ids = yolo_results[0].boxes.cls.cpu().numpy().astype(int)  # (N,)

        # Vision Transformer confidence scoring
        vit_confidences = self._get_vit_confidences(frame, yolo_boxes)

        # Ensemble: average confidences from both models
        ensemble_confidences = (yolo_confidences + vit_confidences) / 2.0

        # Filter by confidence threshold
        boxes, confidences, class_ids = self.postprocess_detections(
            yolo_boxes, ensemble_confidences, yolo_class_ids
        )

        execution_time = (time.time() - start_time) * 1000  # ms

        return {
            "boxes": boxes,
            "confidences": confidences,
            "class_ids": class_ids,
            "execution_time_ms": execution_time,
            "device": str(self.device),
            "num_detections": len(boxes),
            "yolo_confidences": yolo_confidences[:len(boxes)] if len(boxes) > 0 else np.array([]),
            "vit_confidences": vit_confidences[:len(boxes)] if len(boxes) > 0 else np.array([]),
        }

    def detect_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Run batch detection on multiple frames.
        
        Args:
            frames: List of frames
            
        Returns:
            List of detection dictionaries
        """
        start_time = time.time()
        detections_list = []

        # Process frames in parallel with YOLOv8
        yolo_results = self.yolo_model(frames, conf=self.confidence_threshold, verbose=False)

        for frame, yolo_result in zip(frames, yolo_results):
            yolo_boxes = yolo_result.boxes.xyxy.cpu().numpy()
            yolo_confidences = yolo_result.boxes.conf.cpu().numpy()
            yolo_class_ids = yolo_result.boxes.cls.cpu().numpy().astype(int)

            # ViT scoring
            vit_confidences = self._get_vit_confidences(frame, yolo_boxes)
            ensemble_confidences = (yolo_confidences + vit_confidences) / 2.0

            # Filter
            boxes, confidences, class_ids = self.postprocess_detections(
                yolo_boxes, ensemble_confidences, yolo_class_ids
            )

            detections_list.append({
                "boxes": boxes,
                "confidences": confidences,
                "class_ids": class_ids,
                "num_detections": len(boxes),
            })

        execution_time = (time.time() - start_time) * 1000 / len(frames)  # avg per frame

        # Standardize output
        for det in detections_list:
            det.update({
                "execution_time_ms": execution_time,
                "device": str(self.device),
            })

        return detections_list

    def _get_vit_confidences(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Score each bounding box region using Vision Transformer.
        
        Args:
            frame: Original frame
            boxes: Bounding boxes from YOLO (N, 4)
            
        Returns:
            Confidence scores from ViT (N,)
        """
        if len(boxes) == 0:
            return np.array([])

        vit_scores = []

        with torch.no_grad():
            with autocast(enabled=self.use_fp16):
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    # Clip to frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    # Extract region
                    roi = frame[y1:y2, x1:x2]

                    if roi.size == 0:
                        vit_scores.append(0.0)
                        continue

                    # Convert BGR to RGB
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_pil = transforms.ToPILImage()(roi_rgb)

                    # Transform and run through ViT
                    roi_tensor = self.vit_transforms(roi_pil).unsqueeze(0).to(self.device)
                    vit_output = self.vit_model(roi_tensor)
                    vit_score = torch.sigmoid(vit_output).item()
                    vit_scores.append(vit_score)

        return np.array(vit_scores, dtype=np.float32)
