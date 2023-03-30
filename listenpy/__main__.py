import pathlib

from vosk import Model

from .lib_logic import download_and_extract_model, recognize_speech

if __name__ == "__main__":
    print(__file__)
    # Загрузка предобученной модели для русского языка
    path = pathlib.Path(__file__).parent
    model_name = "vosk-model-small-ru-0.22"
    if not (path / model_name).exists():
        print("Модель не установлена.")
        download_and_extract_model(model_name, path)
    model = Model(str(path / model_name))
    # Пример использования функции recognize_speech()
    recognize_speech(model)
