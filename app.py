# app.py
import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf
from datetime import datetime

# ===========================================
# CONFIGURACIÓN
# ===========================================
app = Flask(__name__)

MODEL_PATH = "models/lung_model.tflite"  # Cambiado a TFLite
CLASS_NAMES = ["benigno", "maligno", "normal"]
IMG_SIZE = 224

# ===========================================
# CARGAR MODELO TFLITE
# ===========================================
print("🧠 Cargando modelo TFLite...")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Obtener detalles de entrada/salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("✅ Modelo TFLite cargado correctamente")
    print(f"   📐 Input shape: {input_details[0]['shape']}")
    print(f"   📐 Output shape: {output_details[0]['shape']}")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    exit(1)

# ===========================================
# FUNCIÓN PARA PROCESAR LA IMAGEN
# ===========================================
def preprocess_image(image):
    """Preprocesa la imagen para TFLite"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Redimensionar
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convertir a array y normalizar
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    
    # Agregar dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ===========================================
# FUNCIÓN DE PREDICCIÓN CON TFLITE
# ===========================================
def predict_with_tflite(img_array):
    """Realiza predicción usando TFLite"""
    # Establecer tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Ejecutar inferencia
    interpreter.invoke()
    
    # Obtener predicciones
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    return predictions[0]

# ===========================================
# ENDPOINT DE CLASIFICACIÓN
# ===========================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No se envió ningún archivo"}), 400
        
        file = request.files["file"]
        
        # Validar que sea una imagen
        if file.filename == '':
            return jsonify({"error": "Nombre de archivo vacío"}), 400
        
        # Cargar y procesar imagen
        image = Image.open(io.BytesIO(file.read()))
        processed_img = preprocess_image(image)
        
        # Predicción con TFLite
        preds = predict_with_tflite(processed_img)  # [benigno, maligno, normal]
        
        # Extraer probabilidades
        benigno_pct = float(preds[0] * 100)
        maligno_pct = float(preds[1] * 100)
        normal_pct = float(preds[2] * 100)
        
        # Determinar clase con mayor probabilidad entre las tres
        predicted_idx = int(np.argmax(preds))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(preds[predicted_idx] * 100)

        # Mensaje interpretativo según clase
        if predicted_class == "benigno":
            message = "Se detectó una masa benigna. Monitoreo recomendado."
            alert_level = "warning"
        elif predicted_class == "maligno":
            message = "⚠️ Alerta: posible tumor maligno. Requiere análisis médico inmediato."
            alert_level = "danger"
        else:
            message = "Pulmones normales detectados. No se observan anomalías."
            alert_level = "success"

        
        # Respuesta JSON
        return jsonify({
            "status": "OK",
            "predicted_label": predicted_class,
            "confidence": round(confidence, 2),
            "predictions": {
                "benigno": round(benigno_pct, 2),
                "maligno": round(maligno_pct, 2),
                "normal": round(normal_pct, 2)
            },
            "message": message,
            "alert_level": alert_level,
            "timestamp": datetime.now().isoformat(),
            "model": "TFLite - MobileNetV2"
        })
    
    except Exception as e:
        print(f"❌ Error en /predict: {e}")
        return jsonify({
            "status": "ERROR",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# ===========================================
# ENDPOINT DE SALUD
# ===========================================
@app.route("/health", methods=["GET"])
def health():
    """Endpoint para verificar que la API está funcionando"""
    return jsonify({
        "status": "OK",
        "message": "API funcionando correctamente",
        "model": "TFLite",
        "classes": CLASS_NAMES,
        "image_size": IMG_SIZE,
        "timestamp": datetime.now().isoformat()
    })

# ===========================================
# ENDPOINT DE INFORMACIÓN DEL MODELO
# ===========================================
@app.route("/info", methods=["GET"])
def info():
    """Información del modelo"""
    return jsonify({
        "model_path": MODEL_PATH,
        "model_type": "TensorFlow Lite",
        "architecture": "MobileNetV2 (Transfer Learning)",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "input_size": f"{IMG_SIZE}x{IMG_SIZE}",
        "input_shape": input_details[0]['shape'].tolist(),
        "output_shape": output_details[0]['shape'].tolist(),
        "preprocessing": "Normalización 0-1",
        "timestamp": datetime.now().isoformat()
    })

# ===========================================
# MAIN
# ===========================================
if __name__ == "__main__":
    try:
        from flask_cors import CORS
        CORS(app)
        print("✅ CORS habilitado")
    except ImportError:
        print("⚠️  flask-cors no instalado. Ejecuta: pip install flask-cors")
    
    print("\n" + "=" * 60)
    print("🚀 API DE CLASIFICACIÓN DE CÁNCER PULMONAR")
    print("=" * 60)
    print(f"📍 URL: http://localhost:5000")
    print(f"🧠 Modelo: {MODEL_PATH}")
    print(f"📋 Clases: {CLASS_NAMES}")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)