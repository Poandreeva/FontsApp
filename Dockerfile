# Образ Python
FROM python:3.11-slim

# Установка необходимых системных библиотек
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    pkg-config \
    libhdf5-dev \
    build-essential && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории в контейнере
WORKDIR /app

# Копирование файлов проекта в контейнер
COPY . /app

# Установка зависимостей Python из файла requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Запуск скрипта по умолчанию при старте контейнера
ENTRYPOINT ["python", "3_ModuleRecognition.py"]