# Fonts Recognition App
## Описание
Это консольное приложение для распознавания шрифтов на изображениях. Приложение анализирует изображение, выделяет текст и определяет наиболее вероятный шрифт с указанием вероятности.

## Требования
Для работы приложения необходимо иметь установленный Docker.

# Структура проекта 
(необходимые файлы для работы приложения)
- Dockerfile: Файл для сборки Docker-образа.
- FontsTest.py: Основной скрипт Python, выполняющий распознавание шрифтов.
- font_recognition_model.keras: Обученная модель для распознавания шрифтов.
- label_encoder.pkl: Файл с объектом LabelEncoder.
- requirements.txt: Список библиотек Python, необходимых для работы приложения.

## Установка и запуск
1. Клонируйте репозиторий на свой локальный компьютер:
git clone https://github.com/Poandreeva/FontsApp.git

2. Перейдите в директорию проекта: 
cd <имя_директории>

3. Соберите Docker-образ:
docker build -t fontsapp .

4. Запустите контейнер из образа (замените /path/to/image.png на путь к изображению, на котором необходимо распознать шрифт):
docker run -it --rm fontsapp /app /path/to/image.png
* Пример: docker run -it --rm fontsapp /app /app/dataset/AlumniSansCollegiateOne-Regular/AlumniSansCollegiateOne-Regular_190.png
