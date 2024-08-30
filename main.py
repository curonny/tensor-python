import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

import gradio as gr

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


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


def process_image(image):
    font_path = os.path.join(os.getcwd(), 'font', 'Caveat-Regular.ttf')
    try:
        font = ImageFont.truetype(font_path, size=24)
    except IOError:
        print("No se pudo cargar la fuente, se usará la fuente predeterminada.")
        font = ImageFont.load_default()
    # Preprocesar la imagen para el modelo
    image = image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.array(image, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (input_data - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_ids = np.argsort(output_data[0])[-5:][::-1]  # Obtener las 5 clases con más alta probabilidad
    probabilities = output_data[0][class_ids]
    results = [(labels[class_id], probabilities[i]) for i, class_id in enumerate(class_ids)]
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Definir la fuente y el tamaño del texto
    font = ImageFont.truetype(font_path, size=24)  # Asegúrate de tener esta fuente disponible
    text_height = sum(
        draw.textbbox((0, 0), result[0], font=font)[3] - draw.textbbox((0, 0), result[0], font=font)[1] for result in
        results) + 10 * len(results)

    # Calcular la posición inicial para centrar el texto verticalmente
    start_y = (height - text_height) // 2

    # Dibujar cada resultado en la imagen, centrado horizontalmente
    for i, (label, probability) in enumerate(results):
        text = f"{label}: {probability:.2f}"
        # Obtén el ancho y la altura usando textbbox
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        line_height = text_bbox[3] - text_bbox[1]

        x = (width - text_width) // 2
        y = start_y + i * (line_height + 10)  # 10 es el espacio entre líneas
        draw.text((x, y), text, font=font, fill="white")

    return image


demo = gr.Interface(
    fn=process_image,  # Función que maneja la lógica
    inputs=gr.Image(type="pil", label="Sube tu imagen"),  # Entrada de imagen
    outputs=gr.Image(type="pil", label="Imagen modificada"),  # Salida de imagen modificada
    title="Modificador de Imágenes con TFLite",
    description="Sube una imagen, se procesa con TFLite, y se devuelve modificada."
)

demo.launch()
