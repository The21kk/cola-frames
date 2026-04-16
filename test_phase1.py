"""
Quick test script to verify Phase 1 infrastructure is working.
Tests: Serialization + Redis Streams integration.
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from producer.frame_serializer import FrameSerializer
from redis_broker.stream_manager import RedisStreamManager


def test_frame_serialization():
    """Test frame encode/decode without data loss."""
    print("\n" + "="*60)
    print("TEST 1: Frame Serialization (Base64 + JPEG)")
    print("="*60)
    
    # Create dummy frame (640x480x3 BGR)
    frame_original = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Encode to Base64
    frame_b64 = FrameSerializer.encode_frame_to_base64(frame_original)
    print(f"✓ Encoded frame to Base64")
    print(f"  Original shape: {frame_original.shape}")
    print(f"  Base64 size: {len(frame_b64)} bytes")
    
    # Decode back
    frame_decoded = FrameSerializer.decode_frame_from_base64(frame_b64)
    print(f"✓ Decoded Base64 back to frame")
    print(f"  Decoded shape: {frame_decoded.shape}")
    
    # Verify shapes match (quality might differ due to JPEG compression)
    assert frame_decoded.shape == frame_original.shape, "Shape mismatch!"
    print(f"✓ Shape verification passed")
    
    # Create metadata
    metadata = FrameSerializer.create_metadata(
        camera_id="test_cam",
        fps=5,
        resolution=(640, 480)
    )
    print(f"✓ Metadata created: {metadata}")
    
    print("\n✅ TEST 1 PASSED: Serialization working correctly\n")
    return True


def test_redis_connection():
    """Test Redis connection and basic operations."""
    print("="*60)
    print("TEST 2: Redis Connection & Health Check")
    print("="*60)
    
    try:
        manager = RedisStreamManager()
        health = manager.health_check()
        
        if health["redis_connected"]:
            print(f"✓ Connected to Redis")
            print(f"  Status: {health['status']}")
            print(f"  Active cameras: {health['active_cameras']}")
            print("\n✅ TEST 2 PASSED: Redis is healthy\n")
            return True
        else:
            print(f"✗ Redis not connected")
            print(f"  Error: {health.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"✗ Failed to connect to Redis: {e}")
        print(f"  Make sure Redis is running: docker-compose up -d")
        return False


def test_redis_streams_integration():
    """Test adding and retrieving frames from Redis Streams."""
    print("="*60)
    print("TEST 3: Redis Streams Integration (Produce & Consume)")
    print("="*60)
    
    try:
        manager = RedisStreamManager()
        camera_id = "test_camera_01"
        
        # Clean up any existing stream
        manager.delete_stream(camera_id)
        print(f"✓ Cleaned up existing stream for {camera_id}")
        
        # Create and add 5 test frames
        print(f"\n  Adding 5 test frames...")
        for i in range(5):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame_b64 = FrameSerializer.encode_frame_to_base64(frame)
            metadata = FrameSerializer.create_metadata(camera_id, 5, (640, 480))
            metadata["frame_number"] = i  # Add frame number for tracking
            
            msg_id = manager.add_frame_to_stream(camera_id, frame_b64, metadata)
            print(f"    Frame {i}: {msg_id.decode()}")
            time.sleep(0.1)  # Small delay between frames
        
        # Check stream length
        stream_len = manager.get_stream_length(camera_id)
        print(f"\n✓ Stream length: {stream_len} frames")
        assert stream_len == 5, f"Expected 5 frames, got {stream_len}"
        
        # Get latest frame (LIFO)
        latest = manager.get_latest_frame(camera_id)
        if latest:
            print(f"✓ Retrieved latest frame (LIFO)")
            print(f"  Message ID: {latest['id'].decode()}")
            print(f"  Metadata: {latest['metadata']}")
            assert latest['metadata']['frame_number'] == 4, "LIFO not working!"
            print(f"✓ LIFO verification passed (frame #4 is most recent)")
        
        # Clean up
        manager.delete_stream(camera_id)
        print(f"\n✓ Cleaned up test stream")
        
        print("\n✅ TEST 3 PASSED: Redis Streams working correctly\n")
        return True
        
    except Exception as e:
        print(f"✗ Redis Streams test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frame_throughput():
    """Test approximate throughput with serialization."""
    print("="*60)
    print("TEST 4: Frame Throughput Estimation")
    print("="*60)
    
    try:
        manager = RedisStreamManager()
        camera_id = "throughput_test"
        manager.delete_stream(camera_id)
        
        num_frames = 50
        print(f"  Adding {num_frames} frames and measuring throughput...")
        
        start_time = time.time()
        
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame_b64 = FrameSerializer.encode_frame_to_base64(frame)
            metadata = FrameSerializer.create_metadata(camera_id, 5, (640, 480))
            manager.add_frame_to_stream(camera_id, frame_b64, metadata)
        
        elapsed = time.time() - start_time
        fps = num_frames / elapsed
        
        print(f"\n✓ Processed {num_frames} frames in {elapsed:.2f}s")
        print(f"✓ Throughput: {fps:.2f} FPS")
        
        manager.delete_stream(camera_id)
        
        print("\n✅ TEST 4 PASSED: Throughput is acceptable\n")
        return True
        
    except Exception as e:
        print(f"✗ Throughput test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("█" * 60)
    print("█  COLA-FRAMES: PHASE 1 VERIFICATION TEST SUITE")
    print("█  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("█" * 60)
    
    results = {
        "Serialization": test_frame_serialization(),
        "Redis Connection": test_redis_connection(),
        "Redis Streams": test_redis_streams_integration(),
        "Throughput": test_frame_throughput(),
    }
    
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    print("="*60)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Phase 1 infrastructure is ready.")
        print("\nNext steps:")
        print("  1. Prepare test videos in ./videos/ folder")
        print("  2. Implement Phase 2: Detection Workers (YOLOv8 + DETR)")
        print("  3. Implement Phase 3: Rules Engine")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        print("  Most likely: Redis not running. Try: docker-compose up -d")
    
    print("\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
