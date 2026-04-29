#!/usr/bin/env python
"""Test instantiating GenericDetector."""

import sys
sys.path.insert(0, '.')

print('=== Verificando instalación de librerías ===')
try:
    import ultralytics
    print(f'✓ ultralytics {ultralytics.__version__} instalado')
except ImportError:
    print('✗ ultralytics NO instalado')
    print('  Instalar con: pip install ultralytics>=8.3.0')
    exit(1)

try:
    import torchvision
    print(f'✓ torchvision {torchvision.__version__} instalado')
except ImportError:
    print('✗ torchvision NO instalado')
    print('  Instalar con: pip install torchvision')
    exit(1)

print('\n=== Validación: Instantiating GenericDetector ===')
try:
    from workers.generic_detector import GenericDetector
    
    print('Creando YOLOv10 detector (será descargado en backend si no existe)...')
    print('(Esto puede tomar 1-2 minutos en la primera ejecución)')
    
    # Try to create YOLOv10 detector with CPU (más rápido para prueba)
    detector = GenericDetector(
        model_type='yolov10',
        model_name='yolov10m',  # medium size
        device='cpu',  # Use CPU for faster setup
        batch_size=1,
        confidence_threshold=0.5,
    )
    
    print(f'✓ YOLOv10 detector creado exitosamente')
    print(f'  Model info: {detector.get_model_info()}')
    
except Exception as e:
    print(f'✗ Error creando detector: {e}')
    import traceback
    traceback.print_exc()
    exit(1)

print('\n=== TEST OK ===')
print('Sistema listo para ejecutar tests')
