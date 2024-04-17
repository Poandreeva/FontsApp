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
    Извлекает контуры символов из изображения.

    Args:
        char_image (Image): Изображение, из которого извлекаются контуры.

    Returns:
        list: Список прямоугольников, описывающих границы каждого найденного контура.
    '''
    # Конвертация изображения в массив numpy для обработки
    im_array = np.array(char_image)
    if len(im_array.shape) == 3:
        im_gray = cv2.cvtColor(im_array, cv2.COLOR_RGB2GRAY)  # Перевод в градации серого
    else:
        im_gray = im_array

    # Применение адаптивного порога для улучшения видимости контуров
    im_th = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    # Поиск контуров
    ctrs, _ = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Получение ограничивающих прямоугольников для каждого контура
    contours = [cv2.boundingRect(ctr) for ctr in ctrs]
    return contours

def extract_symbols(char_image, contours):
    '''
    Выделяет и масштабирует символы на основе найденных контуров.

    Args:
        char_image (Image): Исходное изображение, содержащее символы.
        contours (list): Координаты контуров для вырезания символов.

    Returns:
        list: Список изображений символов, преобразованных к стандартному размеру.
    '''
    symbols = []
    for (x, y, w, h) in contours:
        # Обрезка по контуру и масштабирование до размера 28x28
        symbol = char_image.crop((x, y, x + w, y + h))
        symbol = symbol.resize((28, 28))
        symbols.append(np.array(symbol))
    return symbols

def prepare_images(symbols):
    '''
    Нормализует изображения символов для подачи в нейронную сеть.

    Args:
        symbols (list): Список изображений символов.

    Returns:
        numpy.ndarray: Подготовленный массив изображений для модели.
    '''
    # Нормализация и добавление канала для совместимости с архитектурой модели
    processed_images = np.array([np.resize(symbol, (28, 28)) for symbol in symbols])
    processed_images = processed_images / 255.0
    processed_images = processed_images.reshape((len(processed_images), 28, 28, 1))
    return processed_images

def main(assets_dir, image_path):
    '''
    Основная функция для загрузки модели, обработки изображения и вывода предсказаний.

    Args:
        assets_dir (str): Путь к ресурсам (модель и кодировщик меток).
        image_path (str): Путь к изображению для обработки.
    '''
    # Загрузка модели и кодировщика меток
    model = load_model(f"{assets_dir}/font_recognition_model.keras")
    with open(f"{assets_dir}/label_encoder.pkl", 'rb') as file:
        label_encoder = pickle.load(file)
    
    # Обработка изображения
    char_image = Image.open(image_path).convert('L')
    contours = find_contours(char_image)
    symbols = extract_symbols(char_image, contours)
    
    # Подготовка изображений и выполнение предсказаний
    processed_images = prepare_images(symbols)
    predictions = model.predict(processed_images)
    
    # Агрегация предсказаний для определения наиболее вероятного шрифта
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
        print("Usage: fontsapp /app /app/images/your_image.png")
        sys.exit(1)
    assets_dir = sys.argv[1]
    image_path = sys.argv[2]
    main(assets_dir, image_path)

