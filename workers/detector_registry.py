"""
Detector Registry: Dynamic mapping of detector types to classes.

Allows registering new detector implementations without modifying factory code.
Follows the registry pattern for extensibility (open/closed principle).
"""

import logging
from typing import Dict, Type, Any

logger = logging.getLogger(__name__)


class DetectorRegistry:
    """
    Registry of available detector types.
    
    Usage:
        registry = DetectorRegistry()
        registry.register("YOLOVitDetector", YOLOVitDetector)
        detector_class = registry.get("YOLOVitDetector")
    """

    def __init__(self):
        """Initialize empty registry."""
        self._detectors: Dict[str, Type] = {}

    def register(self, name: str, detector_class: Type) -> None:
        """
        Register a detector class.
        
        Args:
            name: String identifier (e.g., "YOLOVitDetector")
            detector_class: Class object that implements BaseDetector
        """
        if name in self._detectors:
            logger.warning(f"Overwriting detector registration: {name}")
        
        self._detectors[name] = detector_class
        logger.debug(f"Registered detector: {name}")

    def get(self, name: str) -> Type:
        """
        Get detector class by name.
        
        Args:
            name: String identifier
            
        Returns:
            Detector class
            
        Raises:
            ValueError: If detector not registered
        """
        if name not in self._detectors:
            available = ", ".join(self._detectors.keys())
            raise ValueError(
                f"Unknown detector type: {name}. "
                f"Available: {available}"
            )
        return self._detectors[name]

    def list_detectors(self) -> list:
        """Get list of registered detector names."""
        return list(self._detectors.keys())

    def is_registered(self, name: str) -> bool:
        """Check if detector type is registered."""
        return name in self._detectors


# Global registry instance
_global_registry = DetectorRegistry()


def register_detector(name: str, detector_class: Type) -> None:
    """
    Register a detector in the global registry.
    
    Args:
        name: String identifier
        detector_class: Detector class
    """
    _global_registry.register(name, detector_class)


def get_detector_class(name: str) -> Type:
    """
    Get detector class from global registry.
    
    Args:
        name: String identifier
        
    Returns:
        Detector class
    """
    return _global_registry.get(name)


def list_registered_detectors() -> list:
    """Get list of all registered detectors."""
    return _global_registry.list_detectors()


# Auto-register built-in detectors on import
def _register_builtin_detectors():
    """
    Auto-register available detector implementations.
    
    Now uses GenericDetector which is model-agnostic and scalable.
    Model type is specified in workers_config.yaml, not via separate detector classes.
    """
    try:
        from workers.generic_detector import GenericDetector
        register_detector("GenericDetector", GenericDetector)
        logger.info("✓ Registered GenericDetector (scalable, model-agnostic)")
    except ImportError as e:
        logger.error(f"Critical: Could not import GenericDetector: {e}")
        raise


# Register built-in detectors on module import
_register_builtin_detectors()
