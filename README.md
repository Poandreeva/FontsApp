# Fonts Recognition App
## Описание
Это консольное приложение для распознавания шрифтов на изображениях. Приложение анализирует изображение, выделяет текст и определяет наиболее вероятный шрифт с указанием вероятности.

## Требования
Для работы приложения необходимо иметь установленный Docker. Текст должен быть написан слева направо горизонтально без наклона и быть загружен в формате png.

# Структура проекта 
## 1. Модуль генерации
- 1_ModuleGeneration.ipynb: Jupyter Notebook, выполняющий генерацию dataset и обучающей и тестовой выборок.
- fonts: Папка, содержащая шрифты для распознавания.

## 2. Модуль обучения нейросети
- 2_ModuleTraining.ipynb: Jupyter Notebook, содержащий модель нейронной сети, ее обучение и валидацию, логгирование значений функции ошибки, вывод основных метрик классификации.
- fonts_dataset.npz: обучающая и тестовая выборка.

### 2.1. Модуль визуализации
- 2_Visualization.ipynb: Jupyter Notebook с кодом для отрисовывки кривых обучающей и тестовой ошибок.
- training_log.csv: Метрики функции потерь и доли правильных ответов в зависимости от эпох обучения.

## 3. Модуль распознавания
(необходимые файлы для работы приложения)
- Dockerfile: Файл для сборки Docker-образа.
- 3_ModuleRecognition.py: Основной скрипт Python, выполняющий распознавание шрифтов.
- font_recognition_model.keras: Обученная модель для распознавания шрифтов.
- label_encoder.pkl: Файл с объектом LabelEncoder.
- requirements.txt: Список библиотек Python, необходимых для работы приложения.

### Установка и запуск
1. Клонируйте репозиторий на свой локальный компьютер:
git clone https://github.com/Poandreeva/FontsApp.git

2. Перейдите в директорию проекта: 
cd <имя_директории>

3. Соберите Docker-образ:
docker build -t fontsapp .

4. Запустите контейнер из образа:

docker run -it --rm \
  -v /home/user/your_image.png:/app/images/your_image.png \
  fontsapp /app /app/images/your_image.png

замените '/home/user/your_image.png' на путь к изображению, на котором необходимо распознать шрифт 

замените 'your_image.png' на имя изображения

* Пример:
  docker run -it --rm \
  -v /Users/PolinaMac/Desktop/GaneshaType-Regular_994.png:/app/images/GaneshaType-Regular_994.png \
  fontsapp /app /app/images/GaneshaType-Regular_994.png

## Дополнительные файлы
- task.txt: ТЗ проекта.
- dataset: база данных со случайными строками.
- training_plots.png: Отрисованные кривые обучающей и тестовой ошибок.
