import base64
import cv2
import numpy as np
from datetime import datetime
from config.settings import FRAME_JPEG_QUALITY, FRAME_FORMAT


class FrameSerializer:
    """
    Handles frame serialization/deserialization for Redis transport.
    Uses Base64 + JPEG compression to minimize bandwidth.
    """

    @staticmethod
    def encode_frame_to_base64(frame: np.ndarray) -> bytes:
        """
        Converts a numpy frame to Base64-encoded JPEG for transmission.
        
        Args:
            frame: OpenCV frame (BGR format, uint8)
            
        Returns:
            Base64-encoded JPEG as bytes
        """
        _, buffer = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY]
        )
        frame_b64 = base64.b64encode(buffer)
        return frame_b64

    @staticmethod
    def decode_frame_from_base64(frame_b64: bytes) -> np.ndarray:
        """
        Converts a Base64-encoded JPEG back to numpy frame.
        
        Args:
            frame_b64: Base64-encoded JPEG bytes
            
        Returns:
            OpenCV frame (BGR format, uint8)
        """
        frame_decoded = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame

    @staticmethod
    def create_metadata(camera_id: str, fps: int, resolution: tuple) -> dict:
        """
        Creates metadata dictionary for a frame.
        
        Args:
            camera_id: Unique camera identifier
            fps: Frames per second
            resolution: Tuple (width, height)
            
        Returns:
            Metadata dictionary
        """
        return {
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
            "fps": fps,
            "resolution": resolution,
            "format": FRAME_FORMAT
        }

    @staticmethod
    def get_frame_size_bytes(frame_b64: bytes) -> int:
        """
        Returns the size in bytes of a Base64-encoded frame.
        
        Args:
            frame_b64: Base64-encoded frame
            
        Returns:
            Size in bytes
        """
        return len(frame_b64)


class FrameDeserializer:
    """
    Wrapper class for frame deserialization operations (alias for FrameSerializer).
    Provides compatibility with code expecting a separate Deserializer class.
    """
    
    @staticmethod
    def decode_frame_from_base64(frame_b64: bytes) -> np.ndarray:
        """
        Converts a Base64-encoded JPEG back to numpy frame.
        
        Args:
            frame_b64: Base64-encoded JPEG bytes
            
        Returns:
            OpenCV frame (BGR format, uint8)
        """
        return FrameSerializer.decode_frame_from_base64(frame_b64)
