import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Cargar el modelo TFLite
model_path = os.path.join(os.getcwd(), 'tfile', 'ssd_mobilenet.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cargar las etiquetas
with open(os.path.join(os.getcwd(), 'tfile', 'ssd_mobilenet.txt'), 'r') as file:
    labels = file.read().splitlines()


def classify_image(image_path):
    # Abrir y procesar la imagen
    image = Image.open(image_path).resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.array(image, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    # Normalizar la imagen
    input_data = (input_data - 127.5) / 127.5  # Normalización como en Flutter

    # Establecer el tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Obtener los resultados
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Procesar los resultados
    class_ids = np.argsort(output_data[0])[-5:][::-1]  # Obtener las 5 clases con más alta probabilidad
    probabilities = output_data[0][class_ids]

    results = [(labels[class_id], probabilities[i]) for i, class_id in enumerate(class_ids)]
    return results


if __name__ == '__main__':
    image_path = os.path.join(os.getcwd(), 'images', 'image4.jpg')  # Cambia la ruta de la imagen según sea necesario
    results = classify_image(image_path)

    for label, probability in results:
        print(f"{label}: {probability:.2f}")
