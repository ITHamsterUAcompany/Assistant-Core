     from vosk import Model, KaldiRecognizer
from settings.model_selector import select_vosk_model, additional_phrases_response, RANDOM_COMMAND_RESPONSES
import pygame
import pyaudio
import logging
import os
import random
from config import models_path
# Ініціалізація модуля mixer з Pygame
pygame.mixer.init()
# Ініціалізація Vosk моделі для розпізнавання мовлення
model = Model(select_vosk_model(models_path))
rec = KaldiRecognizer(model, 8000)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=8000, input=True, frames_per_buffer=8000)
stream.start_stream()

def play_audio(file_path):
    """Відтворює аудіофайл за вказаним шляхом."""
    if not isinstance(file_path, str):
        logging.error(f"Неправильний тип даних: {type(file_path)}. Очікується шлях у форматі рядка.")
        return

    if not os.path.exists(file_path):
        logging.error(f"Файл не знайдено: {file_path}")
        return

    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        logging.info(f"Відтворюється: {file_path}")

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except pygame.error as e:
        logging.error(f"Помилка при відтворенні файлу: {e}")


def play_random_additional_phrases_response():
    """Відтворює випадкову відповідь на додаткові фрази."""
    random_file = random.choice(list(additional_phrases_response))
    play_audio(random_file)


def play_random_response():
    """Відтворює випадкову відповідь з доступних файлів."""
    available_files = [file for file in RANDOM_COMMAND_RESPONSES if os.path.exists(file)]

    if not available_files:
        logging.warning("Відсутні доступні файли для відтворення.")
        return

    random_response = random.choice(available_files)
    play_audio(random_response)