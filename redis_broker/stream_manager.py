import redis
import json
from config.settings import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    STREAM_PREFIX,
    MAX_STREAM_LENGTH
)


class RedisStreamManager:
    """
    Manages Redis Streams for frame ingestion and consumption.
    Implements LIFO strategy: workers always consume the most recent frame.
    """

    def __init__(self):
        """Initialize Redis connection."""
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False
        )
        
        # Test connection
        try:
            self.redis_client.ping()
            print(f"[INFO] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except redis.ConnectionError as e:
            print(f"[ERROR] Failed to connect to Redis: {e}")
            raise

    def add_frame_to_stream(self, camera_id: str, frame_data: bytes, metadata: dict):
        """
        Adds a frame to the Redis Stream for a camera.
        Automatically trims stream to MAX_STREAM_LENGTH (LIFO).
        
        Args:
            camera_id: Unique camera identifier
            frame_data: Frame data in bytes (Base64-encoded JPEG)
            metadata: Dictionary with frame metadata
            
        Returns:
            Message ID from Redis
        """
        stream_key = f"{STREAM_PREFIX}{camera_id}"
        
        message = {
            b"frame": frame_data,
            b"metadata": json.dumps(metadata).encode()
        }
        
        # XADD with automatic trimming (keeps most recent frames)
        message_id = self.redis_client.xadd(
            stream_key,
            message,
            maxlen=MAX_STREAM_LENGTH,
            approximate=False
        )
        
        return message_id

    def get_latest_frame(self, camera_id: str):
        """
        Retrieves the most recent frame from a camera's stream (LIFO).
        This is the key to avoiding accumulated delay.
        
        Args:
            camera_id: Unique camera identifier
            
        Returns:
            Dictionary with 'id', 'frame', 'metadata' or None if stream is empty
        """
        stream_key = f"{STREAM_PREFIX}{camera_id}"
        result = self.redis_client.xrevrange(stream_key, count=1)
        
        if result:
            message_id, data = result[0]
            return {
                "id": message_id,
                "frame": data[b"frame"],
                "metadata": json.loads(data[b"metadata"])
            }
        return None

    def get_stream_length(self, camera_id: str) -> int:
        """
        Returns the number of messages in a camera's stream.
        
        Args:
            camera_id: Unique camera identifier
            
        Returns:
            Number of frames in stream
        """
        stream_key = f"{STREAM_PREFIX}{camera_id}"
        return self.redis_client.xlen(stream_key)

    def delete_stream(self, camera_id: str) -> bool:
        """
        Deletes a camera's stream (useful for cleanup/reset).
        
        Args:
            camera_id: Unique camera identifier
            
        Returns:
            True if stream was deleted, False if it didn't exist
        """
        stream_key = f"{STREAM_PREFIX}{camera_id}"
        result = self.redis_client.delete(stream_key)
        return result > 0

    def get_all_camera_ids(self) -> list:
        """
        Returns list of all active camera IDs (streams).
        
        Returns:
            List of camera IDs
        """
        prefix = STREAM_PREFIX.encode() if isinstance(STREAM_PREFIX, str) else STREAM_PREFIX
        pattern = f"{STREAM_PREFIX}*"
        
        keys = self.redis_client.keys(pattern)
        camera_ids = [
            key.decode().replace(STREAM_PREFIX, '') 
            for key in keys
        ]
        return camera_ids

    def flush_all(self):
        """
        Flushes all data in Redis DB. Use with caution!
        """
        self.redis_client.flushdb()
        print("[WARNING] Redis DB flushed")

    def health_check(self) -> dict:
        """
        Returns health status of Redis connection and stream info.
        
        Returns:
            Dictionary with connection status and stream statistics
        """
        try:
            info = self.redis_client.ping()
            camera_ids = self.get_all_camera_ids()
            
            stream_stats = {}
            for cam_id in camera_ids:
                stream_stats[cam_id] = self.get_stream_length(cam_id)
            
            return {
                "status": "healthy",
                "redis_connected": True,
                "active_cameras": len(camera_ids),
                "stream_statistics": stream_stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "redis_connected": False,
                "error": str(e)
            }
