import json
import os
import pathlib
import zipfile

import pyaudio
import requests
from vosk import KaldiRecognizer, Model

from .lib_hum import replace_words_with_symbols


def download_and_extract_model(model_name: str, path: pathlib.Path):
    """
    Функция скачивания и распаковки модели.

    Входные параметры:
    - model (str): Модель для скачивания.
    - path (str): Путь для сохранения и распаковки модели.
    """
    url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    print(f"Скачивание модели из {url} ...")
    response = requests.get(url, stream=True)
    zip_file_path = path / f"{model_name}.zip"

    print(f"{zip_file_path=}")
    with open(zip_file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Распаковка модели...")
    with zipfile.ZipFile(zip_file_path, "r") as zf:
        zf.extractall(path)

    os.remove(zip_file_path)
    print("Модель успешно скачана и распакована.")


def init_recognizer(model: Model, sample_rate: float):
    """
    Функция инициализации KaldiRecognizer.

    Входные параметры:
    - sample_rate (float): Частота дискретизации микрофона (обычно 16000 Гц).

    Возвращает:
    - KaldiRecognizer: Инициализированный объект KaldiRecognizer.
    """
    return KaldiRecognizer(model, sample_rate)


def recognize_speech(model: Model):
    """
    Функция непрерывного распознавания голоса с микрофона.
    """
    # Инициализируем KaldiRecognizer с частотой дискретизации 16000 Гц
    recognizer = init_recognizer(model, 16000)

    # Создаем объект PyAudio для работы с микрофоном
    p = pyaudio.PyAudio()

    # Открываем поток аудио с параметрами: одноканальный, 16-бит, частота дискретизации 16000 Гц, ввод с микрофона
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=8000,
    )
    stream.start_stream()

    print("\n\n\nГоворите, и текст будет выводиться в реальном времени...")

    i = 0
    try:
        # Цикл непрерывного распознавания голоса
        while True:
            # Считываем аудиоданные из потока микрофона
            data = stream.read(800)

            # Если данных нет, прерываем цикл
            if len(data) == 0:
                break

            # Если аудиоданные достаточны для распознавания, выводим результат в виде текста
            if recognizer.AcceptWaveform(data):
                i += 1
                res = json.loads(recognizer.Result())
                text = replace_words_with_symbols(res["text"])
                if text:
                    print(
                        f"{i}\t> {text}",
                    )
    except KeyboardInterrupt:
        print("\nЗавершение программы")
        exit(0)

    # Закрываем поток аудио и освобождаем ресурсы PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
