import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================
#  CONFIGURACIÓN
# =============================
train_dir = 'dataset/classification/train'
test_dir = 'dataset/classification/test'

img_size = 224
batch_size = 32
epochs = 30  # 🔥 Entrena 30 épocas completas

# =============================
#  CARGAR DATOS
# =============================
print("📁 Cargando datos...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    seed=42
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    seed=42
)

class_names = train_ds.class_names
print(f"✓ Clases encontradas: {class_names}")

# =============================
#  NORMALIZACIÓN Y AUGMENTATION
# =============================
normalization = tf.keras.layers.Rescaling(1./255)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Aplicar normalización
train_ds = train_ds.map(lambda x, y: (normalization(x), y))
test_ds = test_ds.map(lambda x, y: (normalization(x), y))

# Aplicar augmentation solo a train
train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))

# Optimizar performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# =============================
#  CREAR MODELO
# =============================
print("\n🧠 Creando modelo con MobileNetV2...")

model = tf.keras.Sequential([
    # Base preentrenada
    tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    ),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Congelar la base de MobileNetV2 (solo entrenar capas superiores)
model.layers[0].trainable = False

# =============================
#  COMPILAR
# =============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Modelo compilado correctamente")
model.summary()

# =============================
#  CALLBACKS
# =============================
os.makedirs('models', exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'models/best_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
]

# =============================
#  ENTRENAR (sin EarlyStopping)
# =============================
print("\n🚀 Iniciando entrenamiento completo (30 épocas)...")

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
    callbacks=callbacks
)

# =============================
#  GUARDAR MODELOS
# =============================
print("\n💾 Guardando modelo final...")
model.save('models/lung_model_final.h5')
print("✅ Modelo final guardado en 'models/lung_model_final.h5'")

# =============================
#  CONVERTIR A TFLITE
# =============================
print("\n📱 Convirtiendo modelo a TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/lung_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Modelo TFLite guardado en 'models/lung_model.tflite'")

# Tamaños
h5_size = os.path.getsize('models/lung_model_final.h5') / (1024*1024)
tflite_size = os.path.getsize('models/lung_model.tflite') / (1024*1024)
print(f"\n📦 Tamaño H5: {h5_size:.2f} MB")
print(f"📦 Tamaño TFLite: {tflite_size:.2f} MB (reducción: {(1 - tflite_size / h5_size) * 100:.1f}%)")

# =============================
#  EVALUAR
# =============================
print("\n📊 Evaluando modelo en conjunto de prueba...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ Accuracy en test: {test_acc * 100:.2f}%")
print(f"📉 Pérdida (loss) en test: {test_loss:.4f}")

# =============================
#  GRAFICAR RESULTADOS
# =============================
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento', marker='o')
plt.plot(history.history['val_accuracy'], label='Validación', marker='s')
plt.title('Precisión (Accuracy)')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento', marker='o')
plt.plot(history.history['val_loss'], label='Validación', marker='s')
plt.title('Pérdida (Loss)')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/training_results.png', dpi=150)
plt.show()

print("\n✨ ¡Entrenamiento completado con éxito!")
print(f"📈 Mejor accuracy alcanzado: {max(history.history['val_accuracy']) * 100:.2f}%")
