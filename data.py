import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import numpy as np

# =============================
#  CONFIGURACIÓN
# =============================
# Rutas de origen (donde están todas tus imágenes mezcladas)
source_dirs = {
    'benigno': 'dataset/original/benigno',      # 120 imágenes
    'maligno': 'dataset/original/maligno',      # 561 imágenes
    'normal': 'dataset/original/normal'         # 416 imágenes
}

# Ruta de destino
output_base = 'dataset/classification'

# Proporción de división (80% train, 20% test)
train_ratio = 0.80
test_ratio = 0.20

# Semilla para reproducibilidad
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# =============================
#  FUNCIÓN DE ORGANIZACIÓN
# =============================
def organize_dataset(source_dirs, output_base, train_ratio=0.80, test_ratio=0.20):
    """
    Organiza el dataset en estructura train/test para clasificación
    """
    
    # Crear directorios base
    train_dir = os.path.join(output_base, 'train')
    test_dir = os.path.join(output_base, 'test')
    
    # Estadísticas
    stats = {
        'total': {},
        'train': {},
        'test': {}
    }
    
    print("=" * 60)
    print("🗂️  ORGANIZANDO DATASET PARA CLASIFICACIÓN")
    print("=" * 60)
    
    for class_name, source_path in source_dirs.items():
        print(f"\n📁 Procesando clase: {class_name.upper()}")
        
        # Verificar que existe el directorio
        if not os.path.exists(source_path):
            print(f"   ⚠️  ADVERTENCIA: No existe {source_path}")
            continue
        
        # Obtener todas las imágenes
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(Path(source_path).glob(f'*{ext}'))
            all_images.extend(Path(source_path).glob(f'*{ext.upper()}'))
        
        all_images = [str(img) for img in all_images]
        total_images = len(all_images)
        
        if total_images == 0:
            print(f"   ⚠️  No se encontraron imágenes en {source_path}")
            continue
        
        stats['total'][class_name] = total_images
        print(f"   ✓ Encontradas: {total_images} imágenes")
        
        # Dividir en train y test
        train_images, test_images = train_test_split(
            all_images,
            train_size=train_ratio,
            test_size=test_ratio,
            random_state=random_seed,
            shuffle=True
        )
        
        stats['train'][class_name] = len(train_images)
        stats['test'][class_name] = len(test_images)
        
        # Crear directorios de destino
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Copiar imágenes de entrenamiento
        print(f"   📤 Copiando {len(train_images)} imágenes a train/{class_name}...")
        for img_path in train_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(train_class_dir, img_name)
            shutil.copy2(img_path, dest_path)
        
        # Copiar imágenes de test
        print(f"   📤 Copiando {len(test_images)} imágenes a test/{class_name}...")
        for img_path in test_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(test_class_dir, img_name)
            shutil.copy2(img_path, dest_path)
        
        print(f"   ✅ Completado: Train={len(train_images)}, Test={len(test_images)}")
    
    # Mostrar resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DEL DATASET")
    print("=" * 60)
    
    print("\n📈 Distribución por clase:")
    print(f"{'Clase':<15} {'Total':<10} {'Train (80%)':<15} {'Test (20%)':<15}")
    print("-" * 60)
    
    total_all = 0
    total_train = 0
    total_test = 0
    
    for class_name in source_dirs.keys():
        if class_name in stats['total']:
            total = stats['total'][class_name]
            train = stats['train'][class_name]
            test = stats['test'][class_name]
            
            total_all += total
            total_train += train
            total_test += test
            
            print(f"{class_name:<15} {total:<10} {train:<15} {test:<15}")
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {total_all:<10} {total_train:<15} {total_test:<15}")
    
    # Verificar desbalance
    print("\n⚖️  Análisis de balance:")
    if stats['total']:
        max_class = max(stats['total'].values())
        for class_name, count in stats['total'].items():
            ratio = count / max_class
            balance_status = "✓ Balanceado" if ratio > 0.5 else "⚠️  Desbalanceado"
            print(f"   {class_name}: {count}/{max_class} = {ratio:.2%} {balance_status}")
    
    # Sugerencias
    print("\n💡 RECOMENDACIONES:")
    
    if stats['total'].get('benigno', 0) < 200:
        print("   • Clase 'benigno' tiene pocas muestras (120)")
        print("     → Considera usar data augmentation agresivo")
        print("     → Usa class_weight en el entrenamiento")
    
    print("\n📁 Estructura final creada:")
    print(f"""
    {output_base}/
    ├── train/
    │   ├── benigno/     ({stats['train'].get('benigno', 0)} imágenes)
    │   ├── maligno/     ({stats['train'].get('maligno', 0)} imágenes)
    │   └── normal/      ({stats['train'].get('normal', 0)} imágenes)
    └── test/
        ├── benigno/     ({stats['test'].get('benigno', 0)} imágenes)
        ├── maligno/     ({stats['test'].get('maligno', 0)} imágenes)
        └── normal/      ({stats['test'].get('normal', 0)} imágenes)
    """)
    
    print("=" * 60)
    print("✅ ORGANIZACIÓN COMPLETADA")
    print("=" * 60)
    
    return stats

# =============================
#  CALCULAR CLASS WEIGHTS
# =============================
def calculate_class_weights(stats):
    """
    Calcula pesos de clase para balancear el entrenamiento
    """
    print("\n⚖️  CALCULANDO CLASS WEIGHTS")
    print("-" * 60)
    
    train_counts = stats['train']
    total_samples = sum(train_counts.values())
    num_classes = len(train_counts)
    
    class_weights = {}
    
    for idx, (class_name, count) in enumerate(train_counts.items()):
        weight = total_samples / (num_classes * count)
        class_weights[idx] = weight
        print(f"   {class_name}: {weight:.4f}")
    
    print("\n💻 Código para usar en el modelo:")
    print(f"   class_weight = {class_weights}")
    
    return class_weights

# =============================
#  EJECUTAR ORGANIZACIÓN
# =============================
if __name__ == "__main__":
    # Ejecutar organización
    stats = organize_dataset(
        source_dirs=source_dirs,
        output_base=output_base,
        train_ratio=train_ratio,
        test_ratio=test_ratio
    )
    
    # Calcular pesos de clase
    if stats['train']:
        class_weights = calculate_class_weights(stats)
    
    print("\n🚀 ¡Listo! Ahora puedes entrenar tu modelo con:")
    print("   python train_classification.py")