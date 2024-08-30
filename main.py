import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

import gradio as gr

model_path = os.path.join(os.getcwd(), 'tfile', 'ssd_mobilenet.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(os.path.join(os.getcwd(), 'tfile', 'ssd_mobilenet.txt'), 'r') as file:
    labels = file.read().splitlines()


def process_image(image):
    font_path = os.path.join(os.getcwd(), 'font', 'Caveat-Regular.ttf')
    image = image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.array(image, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (input_data - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_ids = np.argsort(output_data[0])[-5:][::-1]
    probabilities = output_data[0][class_ids]
    results = [(labels[class_id], probabilities[i]) for i, class_id in enumerate(class_ids)]
    draw = ImageDraw.Draw(image)
    width, height = image.size

    font = ImageFont.truetype(font_path, size=24)
    text_height = sum(
        draw.textbbox((0, 0), result[0], font=font)[3] - draw.textbbox((0, 0), result[0], font=font)[1] for result in
        results) + 10 * len(results)

    start_y = (height - text_height) // 2

    for i, (label, probability) in enumerate(results):
        text = f"{label}: {probability:.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        line_height = text_bbox[3] - text_bbox[1]
        x = (width - text_width) // 2
        y = start_y + i * (line_height + 10)
        draw.text((x, y), text, font=font, fill="white")

    return image


app = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Select your image"),
    outputs=gr.Image(type="pil", label="Image result"),
    title="Classify image using TFLite",
    description="Classify image using TFLite",
)

app.launch()
