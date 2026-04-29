"""
Detector Factory: Dynamically instantiate detectors from YAML configuration.

Reads worker configuration from YAML file, uses DetectorRegistry to get classes,
and instantiates detector instances with specified parameters.
"""

import os
import logging
from typing import Dict, Any
import yaml

from workers.detector_registry import get_detector_class, list_registered_detectors

logger = logging.getLogger(__name__)


class DetectorFactory:
    """
    Factory for creating detector instances from YAML configuration.
    
    Supports:
    - Loading workers from YAML file
    - Dynamic instantiation using registry
    - Parameter passing and validation
    - Graceful error handling with fallbacks
    """

    def __init__(self, config_path: str):
        """
        Initialize factory with configuration file.
        
        Args:
            config_path: Path to workers_config.yaml
            
        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        self.config_path = config_path
        self.config = self._load_config()
        logger.info(f"Loaded detector configuration from: {config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        
        Returns:
            Dictionary with 'workers' key containing list of worker configs
            
        Raises:
            FileNotFoundError: If file not found
            ValueError: If YAML is invalid
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

        if config is None:
            raise ValueError("Config file is empty")

        if "workers" not in config:
            raise ValueError("Config must contain 'workers' key with list of worker definitions")

        if not isinstance(config["workers"], list):
            raise ValueError("'workers' must be a list")

        return config

    def create_workers(self, use_gpu: bool = True) -> Dict[str, Any]:
        """
        Create all detector instances from configuration.
        
        Uses GenericDetector which loads model based on model_type (yolov10, faster_rcnn, rt_detr).
        Configuration specifies:
        - model_type: Framework/model family (required)
        - model_name: Specific model identifier (required)
        - device: Device string (default: cuda:0)
        - batch_size: Batch size (default: 4)
        - Additional parameters passed to detector
        
        Args:
            use_gpu: Override device setting to use GPU (if not CPU in config)
            
        Returns:
            Dictionary mapping worker_name -> detector_instance
            
        Raises:
            ValueError: If config is invalid or model loading fails
        """
        workers = {}
        available_detectors = list_registered_detectors()
        detector_class = get_detector_class("GenericDetector")

        for worker_config in self.config["workers"]:
            try:
                worker_name = worker_config.get("name")
                model_type = worker_config.get("model_type")
                model_name = worker_config.get("model_name")
                device = worker_config.get("device", "cuda:0" if use_gpu else "cpu")
                batch_size = worker_config.get("batch_size", 4)
                confidence_threshold = worker_config.get("confidence_threshold", 0.5)
                parameters = worker_config.get("parameters", {})

                # Validation
                if not worker_name:
                    raise ValueError("Worker config must have 'name' field")
                if not model_type:
                    raise ValueError("Worker config must have 'model_type' field (e.g., yolov10, faster_rcnn, rt_detr)")
                if not model_name:
                    raise ValueError("Worker config must have 'model_name' field (e.g., yolov10m, fasterrcnn_resnet50_fpn)")

                # Log model info
                logger.info(
                    f"Creating worker: {worker_name} "
                    f"(model: {model_type}/{model_name}, device: {device}, batch: {batch_size})"
                )

                # Instantiate GenericDetector with model parameters
                detector_instance = detector_class(
                    model_type=model_type,
                    model_name=model_name,
                    device=device,
                    batch_size=batch_size,
                    confidence_threshold=confidence_threshold,
                    **parameters,
                )

                workers[worker_name] = detector_instance
                logger.info(f"✓ Worker '{worker_name}' initialized successfully")

            except Exception as e:
                logger.error(f"Failed to create worker '{worker_config.get('name', 'UNKNOWN')}': {e}")
                raise

        logger.info(f"Successfully created {len(workers)} workers from {self.config_path}")
        return workers

    def get_worker_count(self) -> int:
        """Get number of workers defined in config."""
        return len(self.config.get("workers", []))

    def get_config(self) -> Dict[str, Any]:
        """Get loaded configuration dictionary."""
        return self.config


def create_workers_from_config(
    config_path: str,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to create workers in one call.
    
    Args:
        config_path: Path to workers_config.yaml
        use_gpu: Enable GPU acceleration
        
    Returns:
        Dictionary mapping worker_name -> detector_instance
    """
    factory = DetectorFactory(config_path)
    return factory.create_workers(use_gpu=use_gpu)
