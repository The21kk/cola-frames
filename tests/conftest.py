"""
Pytest configuration and shared fixtures for Cola-Frames tests.

Fixtures provided:
- redis_manager: RedisStreamManager instance with cleanup
- sample_frame: Random 480x640 test frame with synthetic object
- video_file: Temporary video file for testing ingestion
- synthetic_video: Synthetic video with 3 moving objects (Phase 2)
- cleanup_redis: Cleanup Redis streams after tests
"""

import pytest
import numpy as np
import cv2
import time
import logging
import tempfile
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from redis_broker.stream_manager import RedisStreamManager
from producer.frame_serializer import FrameSerializer
from config.settings import TEMPORAL_PERSISTENCE_SECONDS


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES: Redis & Stream Management
# ============================================================================

@pytest.fixture(scope="session")
def redis_manager():
    """
    Session-scoped Redis manager.
    Shared across all tests in a session.
    """
    manager = RedisStreamManager()
    yield manager
    # Cleanup at end of session (optional - Redis persists)


@pytest.fixture
def cleanup_redis(redis_manager):
    """
    Cleanup fixture that removes all test streams after each test.
    Use this fixture in tests to auto-cleanup Redis streams.
    """
    yield
    # Cleanup happens after test
    try:
        redis_manager.redis_client.flushdb()
        logger.info("Cleaned up Redis test data")
    except Exception as e:
        logger.warning(f"Failed to cleanup Redis: {e}")


# ============================================================================
# FIXTURES: Frame Data
# ============================================================================

@pytest.fixture
def sample_frame():
    """
    Generate a synthetic test frame (480x640x3 BGR).
    Includes a white rectangle as fake object.
    """
    frame = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
    # Add white rectangle (simulating an object)
    frame[100:200, 150:300] = 255
    return frame


@pytest.fixture
def sample_frames(sample_frame):
    """
    Generate a batch of 5 test frames.
    """
    return [sample_frame.copy() for _ in range(5)]


@pytest.fixture
def frame_metadata():
    """
    Create standard frame metadata.
    """
    return FrameSerializer.create_metadata(
        camera_id="test_camera",
        fps=5,
        resolution=(640, 480)
    )


# ============================================================================
# FIXTURES: Video Files (Temporary)
# ============================================================================

@pytest.fixture
def video_file(tmp_path):
    """
    Create a temporary video file (MP4) for testing frame ingestion.
    
    Specs:
    - Duration: 2 seconds @ 5 FPS = 10 frames
    - Resolution: 640x480
    - Content: Random frames with moving white square
    
    Cleanup: Auto-removed by tmp_path fixture
    """
    video_path = tmp_path / "test_video.mp4"
    
    width, height = 640, 480
    fps = 5
    duration_seconds = 2
    total_frames = fps * duration_seconds
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    try:
        # Generate frames
        for i in range(total_frames):
            frame = np.random.randint(50, 150, (height, width, 3), dtype=np.uint8)
            
            # Add moving white square (simulating person)
            x = int((i / total_frames) * (width - 100))
            y = int((height - 100) / 2)
            cv2.rectangle(frame, (x, y), (x + 100, y + 100), (255, 255, 255), -1)
            
            # Add frame number
            cv2.putText(
                frame,
                f"Frame {i+1}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            out.write(frame)
        
        logger.info(f"Created test video: {video_path} ({total_frames} frames @ {fps} FPS)")
    finally:
        out.release()
    
    yield video_path


@pytest.fixture
def synthetic_video():
    """
    Load real camera video for Phase 2 testing.
    
    Real video specs:
    - File: 2026-04-21-hallmeds.mp4
    - Resolution: 720x1280 (portrait mode)
    - FPS: 12
    - Duration: 38 seconds
    - Content: Real hallway footage with persons
    
    This replaces the synthetic video to test with real camera data.
    """
    # Path to real video in project root
    real_video_path = Path(__file__).parent.parent / "2026-04-21-hallmeds.mp4"
    
    if not real_video_path.exists():
        logger.warning(f"Real video not found at {real_video_path}")
        logger.warning("Falling back to synthetic video generation")
        # Fallback: generate synthetic video if real one not available
        return _generate_fallback_synthetic_video()
    
    logger.info(f"✓ Using real camera video: {real_video_path}")
    logger.info(f"  (720x1280, 12 FPS, 38s, hallway footage)")
    
    yield real_video_path


def _generate_fallback_synthetic_video():
    """Generate synthetic video as fallback if real video not available."""
    import tempfile
    
    tmp_path = Path(tempfile.mkdtemp())
    video_path = tmp_path / "synthetic_phase2.mp4"
    
    frame_width, frame_height = 640, 480
    fps = 12
    duration_seconds = 5
    total_frames = fps * duration_seconds
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_width, frame_height))
    
    try:
        for frame_idx in range(total_frames):
            # Create blank frame with dark background
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            frame[:, :] = (40, 40, 40)
            
            # Red rectangle (moving left-to-right, simulating person)
            x1 = int(50 + (frame_idx / total_frames) * 400)
            y1 = 100
            x2 = x1 + 80
            y2 = y1 + 100
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red
            
            # Green circle (moving top-to-bottom, simulating person)
            cx = 150
            cy = int(50 + (frame_idx / total_frames) * 350)
            cv2.circle(frame, (cx, cy), 40, (0, 255, 0), -1)  # Green
            
            # Blue rectangle (static)
            cv2.rectangle(frame, (400, 200), (550, 350), (255, 0, 0), -1)  # Blue
            
            # Add frame number
            cv2.putText(
                frame,
                f"Frame {frame_idx+1}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            out.write(frame)
        
        logger.info(f"Generated fallback synthetic video: {video_path} ({total_frames} frames @ {fps} FPS)")
    finally:
        out.release()
    
    return video_path


# ============================================================================
# FIXTURES: GPU / Device Management
# ============================================================================

@pytest.fixture(scope="session")
def gpu_available():
    """
    Check if GPU (CUDA) is available in the system.
    Use @pytest.mark.gpu to skip tests on CPU-only systems.
    """
    try:
        import torch
        available = torch.cuda.is_available()
        logger.info(f"GPU available: {available}")
        return available
    except ImportError:
        logger.warning("PyTorch not installed, GPU check skipped")
        return False


@pytest.fixture
def cleanup_gpu():
    """
    Cleanup GPU memory after tests.
    Use this fixture in GPU-intensive tests.
    """
    yield
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cleared")
    except Exception as e:
        logger.warning(f"GPU cleanup failed: {e}")


# ============================================================================
# PYTEST MARKERS
# ============================================================================

def pytest_configure(config):
    """
    Register custom pytest markers.
    """
    config.addinivalue_line(
        "markers", "gpu: mark test as GPU-dependent (skip if no CUDA)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (useful for CI pipelines)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


# ============================================================================
# PYTEST HOOKS
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Auto-skip GPU tests if CUDA not available.
    """
    try:
        import torch
        skip_gpu = not torch.cuda.is_available()
    except ImportError:
        skip_gpu = True
    
    if skip_gpu:
        skip_marker = pytest.mark.skip(reason="GPU/CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_marker)
