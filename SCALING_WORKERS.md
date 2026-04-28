# Scaling Workers to N Instances

## Overview

El nuevo sistema de **workers escalable** permite agregar, remover y configurar detectores dinámicamente sin tocar el código. Todo se define en un archivo YAML.

## File Structure

```
config/
├── settings.py                 ← Nuevo: WORKERS_CONFIG_PATH
└── workers_config.yaml         ← NUEVO: Configuración de workers (sin hardcode)

workers/
├── detector_registry.py        ← NUEVO: Registro dinámico de tipos
├── detector_factory.py         ← NUEVO: Factory que lee YAML e instancia workers
├── worker_pool.py              ← REFACTORIZADO: Usa factory en lugar de hardcode
├── base_detector.py
├── yolo_vit_detector.py
└── frcnn_rtdetr_detector.py
```

## How to Scale

### Current Setup (2 workers)

```yaml
# config/workers_config.yaml
workers:
  - name: "yolo_vit_1"
    type: "YOLOVitDetector"
    device: "cuda:0"
    batch_size: 4

  - name: "frcnn_rtdetr_1"
    type: "FasterRCNNRtdetrDetector"
    device: "cuda:0"
    batch_size: 2
```

### Scale to 3 Workers (Same Type, Different GPUs)

```yaml
workers:
  - name: "yolo_vit_1"
    type: "YOLOVitDetector"
    device: "cuda:0"
    batch_size: 4

  - name: "yolo_vit_2"
    type: "YOLOVitDetector"
    device: "cuda:1"          # ← Different GPU
    batch_size: 4

  - name: "frcnn_rtdetr_1"
    type: "FasterRCNNRtdetrDetector"
    device: "cuda:0"
    batch_size: 2
```

### Scale to 5 Workers (Multiple Types, Multi-GPU)

```yaml
workers:
  # GPU 0: YOLO instances
  - name: "yolo_vit_1"
    type: "YOLOVitDetector"
    device: "cuda:0"
    batch_size: 4

  - name: "yolo_vit_2"
    type: "YOLOVitDetector"
    device: "cuda:0"          # Same GPU, different instance
    batch_size: 4

  # GPU 1: FRCNN instances  
  - name: "frcnn_rtdetr_1"
    type: "FasterRCNNRtdetrDetector"
    device: "cuda:1"
    batch_size: 2

  - name: "frcnn_rtdetr_2"
    type: "FasterRCNNRtdetrDetector"
    device: "cuda:1"
    batch_size: 2

  # GPU 2: CPU fallback (lightweight)
  - name: "lightweight_1"
    type: "YOLOVitDetector"
    device: "cpu"
    batch_size: 1
```

### Add New Detector Type

1. **Implement** new detector class (inherits from `BaseDetector`):

```python
# workers/my_detector.py
from workers.base_detector import BaseDetector

class MyCustomDetector(BaseDetector):
    def __init__(self, device="cuda:0", batch_size=4, **kwargs):
        super().__init__(
            model_name="my_detector",
            device=device,
            batch_size=batch_size,
        )
        # Load your models...
    
    def detect(self, frame):
        # Your detection logic
        return {
            "boxes": [...],
            "confidences": [...],
            "class_ids": [...],
            "num_detections": len(boxes),
            "execution_time_ms": 0,
        }
    
    def detect_batch(self, frames):
        # Batch version
        return [self.detect(f) for f in frames]
```

2. **Register** in detector registry (happens automatically on import if added to the auto-registration):

```python
# workers/detector_registry.py - already done, but for new detectors:
def _register_builtin_detectors():
    # ... existing registrations ...
    try:
        from workers.my_detector import MyCustomDetector
        register_detector("MyCustomDetector", MyCustomDetector)
    except ImportError:
        logger.warning("Could not import MyCustomDetector")
```

3. **Use** in YAML config:

```yaml
workers:
  - name: "custom_1"
    type: "MyCustomDetector"    # ← Your new detector type
    device: "cuda:2"
    batch_size: 3
    parameters:
      confidence_threshold: 0.6
```

## Key Design Patterns

### 1. **Registry Pattern** (detector_registry.py)
- Centralized mapping of detector names → classes
- Extensible without modifying factory code
- Used by factory to instantiate workers

### 2. **Factory Pattern** (detector_factory.py)
- Reads YAML configuration
- Uses registry to get detector classes
- Creates worker instances with specified parameters
- Validates configuration before instantiation

### 3. **Ensemble Approach**
- Each camera → **all workers** process it
- Workers publish detections independently
- Enables ensemble voting at rules engine level

## Usage in Code

### Before (Hardcoded)
```python
from workers.worker_pool import WorkerPool

pool = WorkerPool(num_workers=2, use_gpu=True)
pool.start()
```

### After (Scalable)
```python
from workers.worker_pool import WorkerPool
from config.settings import WORKERS_CONFIG_PATH

# Auto-loads from config/workers_config.yaml
pool = WorkerPool(config_path=WORKERS_CONFIG_PATH)
pool.start()
```

## Configuration Override with Environment Variables

```bash
# Set custom config path via env var
export WORKERS_CONFIG_PATH=/path/to/custom_workers.yaml

# WorkerPool will use this instead of default
python your_app.py
```

## Monitoring Scaling

```python
# Get status of all workers
status = pool.get_worker_status()
print(f"Active workers: {len(pool.workers)}")
for worker_name, worker_status in status.items():
    print(f"  {worker_name}: {worker_status}")
```

## Example: Scale from 2 to 4 Workers

### Step 1: Edit config/workers_config.yaml
```yaml
workers:
  - name: "yolo_vit_1"
    type: "YOLOVitDetector"
    device: "cuda:0"
    batch_size: 4

  - name: "yolo_vit_2"              # ← NEW
    type: "YOLOVitDetector"
    device: "cuda:1"
    batch_size: 4

  - name: "frcnn_rtdetr_1"
    type: "FasterRCNNRtdetrDetector"
    device: "cuda:0"
    batch_size: 2

  - name: "frcnn_rtdetr_2"          # ← NEW
    type: "FasterRCNNRtdetrDetector"
    device: "cuda:1"
    batch_size: 2
```

### Step 2: Restart WorkerPool
```python
pool = WorkerPool()  # Loads new config automatically
pool.start()
# Now 4 workers are processing all cameras
```

## Removed Code

The following hardcoded worker initialization code has been removed:

### ❌ OLD (worker_pool.py)
```python
from workers.yolo_vit_detector import YOLOVitDetector
from workers.frcnn_rtdetr_detector import FasterRCNNRtdetrDetector

def _initialize_workers(self):
    self.workers["yolo_vit"] = YOLOVitDetector(
        device=DEVICE_WORKER_1,
        batch_size=self.batch_size,
        use_fp16=True,
    )
    self.workers["frcnn_rtdetr"] = FasterRCNNRtdetrDetector(
        device=DEVICE_WORKER_2,
        batch_size=self.batch_size,
        use_fp16=True,
    )
```

### ✅ NEW (worker_pool.py)
```python
from workers.detector_factory import DetectorFactory

def _initialize_workers(self):
    factory = DetectorFactory(self.config_path)
    self.workers = factory.create_workers(use_gpu=self.use_gpu)
```

## Benefits

| Feature | Before | After |
|---------|--------|-------|
| **Number of workers** | Hardcoded (2) | Any N (YAML) |
| **Code changes to scale** | Modify Python code | Edit YAML ✓ |
| **New detector type** | Modify factory | Register in registry ✓ |
| **Multi-GPU support** | Limited | Full (per-worker device) ✓ |
| **Worker isolation** | Coupled | Independent configs ✓ |
| **Runtime changes** | Restart app | Edit config + restart ✓ |

## Testing

Run validation tests:
```bash
python test_scalable_architecture.py
```

Output shows:
- Registry status (detector types available)
- Configuration loading (workers from YAML)
- Instantiation validation (can create each worker)
- Scalability benefits

## Next Steps

1. ✅ **Done**: Scalable architecture (YAML-based, N workers)
2. ✅ **Done**: Remove hardcoded workers
3. 📋 **Optional**: Add monitoring dashboard for worker status
4. 📋 **Optional**: Add dynamic worker reloading (without restart)
5. 📋 **Optional**: Add worker health checks and auto-recovery
