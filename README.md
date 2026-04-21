# Cola-Frames: Detección de Personas para Videovigilancia

## Descripción

Sistema distribuido de visión computacional para validar detecciones de movimiento mediante un modelo ensemble (YOLOv8 + DETR) en un centro de monitoreo con 600 cámaras.

**Fase Actual**: Infraestructura Core (Redis Streams + Ingesta RTSP)

### Arquitectura

```
RTSP Stream
    ↓
Producer (RTSPIngester)
    ↓
Redis Streams (camera:*)
    ↓
Workers (Detection Ensemble)
    ↓
Rules Engine (Temporal + ROI)
    ↓
Alerts
```

---

## Instalación

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (para Redis)
- GPU (opcional, para inference más rápido)

### Setup

```bash
# 1. Clonar/copiar proyecto
cd /home/nicolas/Desktop/U/cola-frames

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env si es necesario (redis host/port)

# 5. Iniciar Redis
docker-compose up -d

# Verificar Redis
redis-cli ping  # Debería responder "PONG"
```

---

## Testing & Validación

Cola-Frames incluye una suite completa de tests automatizados que validan toda la infraestructura.

### Ejecutar Tests

#### Phase 1 Tests (Serialización + Redis + Ingesta)
```bash
# CPU-only, sin GPU requerido (~5 minutos)
pytest tests/test_phase1_infrastructure.py -v

# Verificar recopilación de tests
pytest tests/test_phase1_infrastructure.py --collect-only -q
```

**Coverage**: 13 tests
- Frame serialization (encode/decode, compression)
- Redis connection & health check
- RTSP/video file ingestion
- Stream monitoring (real-time arrivals)
- Frame retrieval & LIFO consumption
- Throughput analysis (100+ FPS)

#### Phase 2 Tests (Detectores + Consenso + Alertas)
```bash
# Requiere GPU (CUDA). Si no hay GPU, los tests se skipean
pytest tests/test_phase2_detection.py -v

# Solo CPU tests (sin GPU)
pytest tests/test_phase2_detection.py -v -m "not gpu"

# Con mark de GPU
pytest tests/test_phase2_detection.py -v -m "gpu"
```

**Coverage**: 26 tests
- GPU/CUDA availability & device selection
- YOLOv8+Vision Transformer detector
- Faster R-CNN+RT-DETR detector
- Batch processing
- Consensus voting (IoU matching)
- Temporal filtering & persistence
- ROI validation (inclusion/exclusion)
- Alert generation & severity
- Detection store
- Performance analysis
- Alert statistics

#### Phase 2 End-to-End Pipeline Tests
```bash
# Pipeline completo frame→detect→consensus→alerts
pytest tests/test_phase2_end_to_end.py -v

# Solo tests rápidos (skip @slow)
pytest tests/test_phase2_end_to_end.py -v -m "not slow"
```

**Coverage**: 10 tests
- Single frame pipeline
- Multi-frame pipeline (video)
- Pipeline robustness (empty/mixed detections)
- Performance analysis (latency, throughput)
- Memory stability

### Ejecutar Todos los Tests

```bash
# Todos los tests (GPU tests se skipean si no CUDA)
pytest tests/ -v

# Con cobertura
pytest tests/ -v --cov=. --cov-report=html

# Solo CPU-compatible
pytest tests/ -v -m "not gpu"

# Recolectar sin ejecutar
pytest tests/ --collect-only -q
```

### Estructura de Tests

```
tests/
├── conftest.py                   # Fixtures compartidas
│   ├── redis_manager
│   ├── sample_frame / sample_frames
│   ├── video_file (temporal)
│   ├── synthetic_video (3 objetos)
│   ├── gpu_available
│   └── cleanup_gpu
│
├── test_phase1_infrastructure.py
│   ├── TestFrameSerialization (3 tests)
│   ├── TestRedisConnection (2 tests)
│   ├── TestRTSPIngestion (2 tests)
│   ├── TestStreamMonitoring (1 test)
│   ├── TestStreamRetrieval (1 test)
│   ├── TestLIFOConsumption (1 test)
│   ├── TestThroughput (2 tests)
│   └── TestPhase1Integration (1 test)
│
├── test_phase2_detection.py
│   ├── TestGPUSupport (4 tests, @gpu)
│   ├── TestDetectionWorkers (5 tests, @gpu)
│   ├── TestConsensusVoting (3 tests)
│   ├── TestTemporalFiltering (1 test)
│   ├── TestROIValidation (2 tests)
│   ├── TestAlertGeneration (3 tests)
│   ├── TestDetectionStore (2 tests)
│   ├── TestPerformance (2 tests, @gpu)
│   ├── TestAlerting (2 tests)
│   └── TestPhase2Integration (2 tests)
│
└── test_phase2_end_to_end.py
    ├── TestSingleFramePipeline (2 tests, @gpu)
    ├── TestMultiFramePipeline (2 tests, @gpu @slow)
    ├── TestPipelineRobustness (3 tests, @gpu)
    ├── TestPipelinePerformance (2 tests, @gpu @slow)
    └── TestFullValidation (1 test, @gpu)
```

### Test Markers

```bash
# GPU tests (skip si no CUDA disponible)
pytest tests/ -m gpu

# Slow tests (~1-5s por test)
pytest tests/ -m slow

# Integration tests
pytest tests/ -m integration

# CPU-only (skip GPU tests)
pytest tests/ -m "not gpu"
```

### Ejemplo de Uso del Sistema (Development)

```python
from producer.rtsp_ingester import RTSPIngester
from redis_broker.stream_manager import RedisStreamManager
import time

# Simular ingesta con video local
ingester = RTSPIngester(
    camera_id="cam_test_01",
    rtsp_url="phase2_test_video.mp4",  # O: "rtsp://192.168.1.100:554/stream"
    fps_target=5
)
ingester.start()

# Esperar ingesta
time.sleep(5)

# Verificar frames en Redis
manager = RedisStreamManager()
frames_count = manager.get_stream_length("cam_test_01")
print(f"Frames en stream: {frames_count}")

# Obtener frame más reciente
latest = manager.get_latest_frame("cam_test_01")
if latest:
    print(f"Frame ID: {latest['id']}")
    print(f"Metadata: {latest['metadata']}")

ingester.stop()

# Para Phase 2 (GPU), ver tests/test_phase2_end_to_end.py
```

---

## Estructura del Proyecto

```
cola-frames/
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuración centralizada
│
├── producer/
│   ├── __init__.py
│   ├── frame_serializer.py      # Encode/decode frames (Base64 + JPEG)
│   └── rtsp_ingester.py         # Ingesta RTSP → Redis Streams
│
├── redis_broker/
│   ├── __init__.py
│   └── stream_manager.py        # Redis Streams manager
│
├── workers/                      # (Próxima fase)
│   └── __init__.py
│
├── rules_engine/                 # (Próxima fase)
│   └── __init__.py
│
├── tests/
│   └── __init__.py
│
├── requirements.txt
├── .env
├── .env.example
├── docker-compose.yml
└── README.md
```

---

## Configuración

### config/settings.py

| Variable | Default | Descripción |
|----------|---------|-------------|
| `REDIS_HOST` | localhost | Host de Redis |
| `REDIS_PORT` | 6379 | Puerto de Redis |
| `MAX_STREAM_LENGTH` | 1000 | Máximo frames en stream (LIFO trimming) |
| `FRAME_JPEG_QUALITY` | 80 | Calidad JPEG (0-100) |
| `TEMPORAL_PERSISTENCE_SECONDS` | 3 | Segundos para persistencia temporal |

---

## Decisiones de Diseño - Fase 1

| Aspecto | Decisión | Justificación |
|--------|----------|---------------|
| **Serialización** | Base64 + JPEG (calidad 80) | Balance latencia vs calidad; fácil de transportar en JSON |
| **Consumo Redis** | XREVRANGE (LIFO) | Evita retrasos acumulados; siempre procesa frame más reciente |
| **Trimming** | maxlen=1000 + approximate=False | Limita memory; ~1 segundo de buffer a 1 FPS |
| **Threading** | Daemon threads | Simple, suficiente para ingesta |

---

## Próximas Fases

### Fase 2: Workers de Detección (2 días)
- YOLOv8 detector
- DETR detector
- Ensemble logic
- Worker pool orchestrator

### Fase 3: Motor de Reglas (1 día)
- Temporal state persistence
- ROI filtering
- Alert engine

### Fase 4: Testing & Demo (1 día)
- Test suite
- Demo local video
- Documentation

---

## Troubleshooting

### Redis no se conecta
```bash
# Verificar Redis está corriendo
docker-compose ps

# Reiniciar Redis
docker-compose restart redis

# Ver logs
docker-compose logs redis
```

### Frames no se guardan en Redis
```bash
# Verificar stream
redis-cli XLEN camera:cam_test_01

# Ver último frame
redis-cli XREVRANGE camera:cam_test_01 COUNT 1
```

### Video local no abre
- Asegúrate que el archivo está en la ruta correcta
- OpenCV soporta: `.mp4`, `.avi`, `.mov`, `.mkv`
- Ejemplo: `"./videos/test.mp4"` o ruta absoluta

---

## Autores

- **Nicolás Schofield** - Taller Profesional 2026, Universidad Diego Portales

---

## Licencia

Proyecto académico - Uso interno
