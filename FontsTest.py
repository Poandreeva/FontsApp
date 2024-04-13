#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
import pickle

def find_contours(char_image):
    '''Определение и возврат контуров символов'''
    im_array = np.array(char_image)
    if len(im_array.shape) == 3:
        im_gray = cv2.cvtColor(im_array, cv2.COLOR_RGB2GRAY)
    else:
        im_gray = im_array

    im_th = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    ctrs, _ = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.boundingRect(ctr) for ctr in ctrs]
    return contours

def extract_symbols(char_image, contours):
    '''Извлечение символов из изображения на основе данных контуров'''
    symbols = []
    for (x, y, w, h) in contours:
        symbol = char_image.crop((x, y, x + w, y + h))
        symbol = symbol.resize((28, 28))
        symbols.append(np.array(symbol))
    return symbols

def prepare_images(symbols):
    '''Подготовка изображений символов для модели'''
    processed_images = np.array([np.resize(symbol, (28, 28)) for symbol in symbols])
    processed_images = processed_images / 255.0
    processed_images = processed_images.reshape((len(processed_images), 28, 28, 1))
    return processed_images

def main(assets_dir, image_path):
    '''Загрузка модели и предсказание шрифта для изображения'''
    model = load_model(f"{assets_dir}/font_recognition_model.keras")
    with open(f"{assets_dir}/label_encoder.pkl", 'rb') as file:
        label_encoder = pickle.load(file)
    
    char_image = Image.open(image_path).convert('L')
    contours = find_contours(char_image)
    symbols = extract_symbols(char_image, contours)
    processed_images = prepare_images(symbols)

    predictions = model.predict(processed_images)
    print('Вероятность принадлежности к шрифту:')
    for i, prediction in enumerate(predictions):
        predicted_class = np.argmax(prediction)
        predicted_probability = prediction[predicted_class]
        predicted_font = label_encoder.inverse_transform([predicted_class])[0]
        print(f"Символ {i + 1}: {predicted_font} ({predicted_probability:.4f})")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python FontsTest.py <assets_directory> <image_path>")
        sys.exit(1)
    assets_dir = sys.argv[1]
    image_path = sys.argv[2]
    main(assets_dir, image_path)

