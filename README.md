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

## Uso - Fase 1 (Testing Local)

### Opción A: Con video local (recomendado para MVP)

```python
from producer.rtsp_ingester import RTSPIngester
from redis_broker.stream_manager import RedisStreamManager

# Simular ingesta con video local
ingester = RTSPIngester(
    camera_id="cam_test_01",
    rtsp_url="video.mp4",  # OpenCV soporta archivos locales como "RTSP"
    fps_target=5
)
ingester.start()

# Esperar unos segundos...
import time
time.sleep(10)

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
```

### Opción B: Con stream RTSP real

```python
from producer.rtsp_ingester import RTSPIngester

ingester = RTSPIngester(
    camera_id="cam_tienda",
    rtsp_url="rtsp://192.168.1.100:554/stream",
    fps_target=5
)
ingester.start()
# El ingester corre en background...
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
