#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import cv2
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


'Поиск контура символа'
def find_contours(char_image):
    
    im_array = np.array(char_image)
    if len(im_array.shape) == 3:
        im_gray = cv2.cvtColor(im_array, cv2.COLOR_RGB2GRAY)
    else:
        im_gray = im_array

    # Функция адаптивного порога 'adaptiveThreshold' использована для улучшения видимости контуров
    im_th = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ctrs, _ = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.boundingRect(ctr) for ctr in ctrs]
    
    return contours


'Выделение символа из строки'
def extract_symbols(char_image, contours):
    
    symbols = []
    for (x, y, w, h) in contours:
        symbol = char_image.crop((x, y, x + w, y + h))
        symbol = symbol.resize((28, 28))
        symbols.append(np.array(symbol))
        
    return symbols


'Нормализация и добавление канала для совместимости с архитектурой модели'
def prepare_images(symbols):
    processed_images = np.array([np.resize(symbol, (28, 28)) for symbol in symbols])
    processed_images = processed_images / 255.0
    processed_images = processed_images.reshape((len(processed_images), 28, 28, 1))
    return processed_images


def main(assets_dir, image_path):
    
    model = load_model(f"{assets_dir}/font_recognition_model.keras")
    with open(f"{assets_dir}/label_encoder.pkl", 'rb') as file:
        label_encoder = pickle.load(file)
    
    char_image = Image.open(image_path).convert('L')
    contours = find_contours(char_image)
    symbols = extract_symbols(char_image, contours)
    
    processed_images = prepare_images(symbols)
    predictions = model.predict(processed_images)
    
    font_count = {}
    total_probability = 0
    for prediction in predictions:
        predicted_class = np.argmax(prediction)
        predicted_probability = prediction[predicted_class]
        predicted_font = label_encoder.inverse_transform([predicted_class])[0]
        font_count[predicted_font] = font_count.get(predicted_font, 0) + 1
        total_probability += predicted_probability

    average_probability = total_probability / len(predictions)
    most_common_font = max(font_count, key=font_count.get)

    print(f"Средняя вероятность по символам: {average_probability:.4f}")
    print(f"Наиболее вероятный шрифт: {most_common_font}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Использование: fontsapp /app /app/images/your_image.png")
        sys.exit(1)
    assets_dir = sys.argv[1]
    image_path = sys.argv[2]
    main(assets_dir, image_path)