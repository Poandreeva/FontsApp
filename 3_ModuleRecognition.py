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
    '''
    Поиск контуров в изображении для выделения отдельных символов.
    
    Args:
    char_image (Image): Изображение, из которого необходимо извлечь контуры.
    
    Returns:
    list: Список прямоугольников, описывающих найденные контуры.
    '''
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
    '''
    Выделение символов из строки по найденным контурам.
    
    Args:
    char_image (Image): Изображение строки.
    contours (list): Список контуров символов.
    
    Returns:
    list: Список изображений выделенных символов.
    '''
    symbols = []
    for (x, y, w, h) in contours:
        symbol = char_image.crop((x, y, x + w, y + h))
        symbol = symbol.resize((28, 28))
        symbols.append(np.array(symbol))
    return symbols


def prepare_images(symbols):
    '''
    Подготовка изображений символов для модели.
    
    Args:
    symbols (list): Список изображений выделенных символов, которые необходимо .
    
    Returns:
    numpy.ndarray: Массив numpy изображений размером 28x28 пикселей в диапазоне от 0 до 1 с добавлением дополнительного измерения для канала.
    '''
    processed_images = np.array([np.resize(symbol, (28, 28)) for symbol in symbols])
    processed_images = processed_images / 255.0
    processed_images = processed_images.reshape((len(processed_images), 28, 28, 1))
    return processed_images

def main(assets_dir, image_path):
    '''
    Загрузка модели и предсказание типа шрифта на изображении.
    
    Args:
        assets_dir (str): Директория с моделью и другими ресурсами.
        image_path (str): Путь к файлу изображения для распознавания шрифта.
    
    Шаги:
        - Загружает обученную модель и кодировщик меток из указанной директории.
        - Открывает указанное изображение, преобразует его в оттенки серого и находит контуры.
        - Извлекает символы из контуров и предварительно обрабатывает их.
        - Совершает предсказания на этих предварительно обработанных изображениях с помощью загруженной модели.
        - Агрегирует предсказания для определения наиболее вероятного шрифта и его средней вероятности предсказания.
    '''

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
        print("Usage: python FontsTest.py <assets_directory> <image_path>")
        sys.exit(1)
    assets_dir = sys.argv[1]
    image_path = sys.argv[2]
    main(assets_dir, image_path)

